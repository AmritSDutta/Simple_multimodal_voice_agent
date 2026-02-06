import asyncio
import re
from dataclasses import dataclass, field

from agents import Agent, Runner
from google import genai
from google.genai import types
from google.genai.client import AsyncClient
from google.genai.types import GenerateContentResponse
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing_extensions import TypedDict

SYSTEM_PROMPT = """
You are an autonomous Tic-Tac-Toe agent.
GENERAL RULES:
1. You play exactly one move per turn.
2. A move must target a cell that currently contains '.' (empty).
3. Choose the strongest legal move for {{SYMBOL}} only.
4. Never choose a filled cell.
5. Never describe reasoning, analysis, or commentary.


OBJECTIVE:
Maximize your chance of winning and minimize opponent advantage.

WIN-MAXIMIZATION STRATEGY (APPLY IN ORDER):
1. Immediate Win: play any move that wins instantly.
2. Block Opponent: if opponent can win next turn, block that move.
3. Center: take (1,1) if empty.
4. Corners: take any available corner.
5. Best Available: choose the most advantageous remaining empty cell.

RULES:
- You must pick exactly one empty cell.
- Never select a filled square.
- No explanations.

OUTPUT FORMAT (STRICT):
Return only:

    row,col

No other text, punctuation, or formatting.
"""

PLAYER_TEMPLATE = """
make your move.

YOUR SYMBOL: {{SYMBOL}}
BOARD STATE:
{{BOARD}}

OUTPUT FORMAT (STRICT):
Return only:

    row,col

No other text, punctuation, or formatting.
"""


def _parse_coord(s: str) -> tuple[int, int] | None:
    m = re.search(r"(-?\d+)\s*[, ]\s*(-?\d+)", s)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _valid_move(board, i, j):
    return 0 <= i < 3 and 0 <= j < 3 and board[i][j] == '.'


def _check_winner(b):
    lines = ([(r, c) for c in range(3)] for r in range(3))  # rows generator
    wins = []
    for r in range(3):
        if b[r][0] == b[r][1] == b[r][2] != '.':
            return b[r][0]
    for c in range(3):
        if b[0][c] == b[1][c] == b[2][c] != '.':
            return b[0][c]
    if b[0][0] == b[1][1] == b[2][2] != '.':
        return b[0][0]
    if b[0][2] == b[1][1] == b[2][0] != '.':
        return b[0][2]
    if all(cell != '.' for row in b for cell in row):
        return 'DRAW'
    return None


player_one_agent = Agent(
    name='player_one',
    instructions=SYSTEM_PROMPT,
    model="gpt-4o-mini"
)

_genai_client: AsyncClient = genai.Client().aio
player_two_agent = _genai_client.chats.create(
    model='gemini-2.5-flash',
    config=types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
    )
)


class Context(TypedDict):
    game_id: str


# Use dataclass for state with default values
@dataclass
class State:
    moves: tuple[int, int] = None
    game_status: str = None
    board: list[list[str]] = field(default_factory=lambda: [['.' for _ in range(3)] for _ in range(3)])
    last_player: str = None

    def print_box(self):
        for row in self.board:
            line = " ".join("   " if c == "." else f" {c} " for c in row)
            print(f"|{line}|")

    def valid(self, i, j):
        return 0 <= i < 3 and 0 <= j < 3 and self.board[i][j] == '.'


async def coordinator_node(state: State):
    if state.last_player is None:
        return Command(
            update={"board": state.board},
            goto="player_one_node"  # Route directly here as game begis
        )

    i, j = state.moves
    symbol = state.last_player
    # Validate move
    if not _valid_move(state.board, i, j):
        return Command(
            update={"game_status": f"Invalid move {(i, j)} by {symbol}"},
            goto=END
        )

    state.board[i][j] = symbol  # Apply move
    state.print_box()

    win = _check_winner(state.board)  # Check win conditions
    if win == 'DRAW':
        return Command(
            update={"game_status": f"DRAW. Board: {state.board}"},
            goto=END
        )
    if win in ('X', 'O'):
        return Command(
            update={"game_status": f"WINNER {win}. Final: {state.board}"},
            goto=END
        )

    # Determine next player
    next_player = "player_two_node" if state.last_player == 'O' else "player_one_node"
    return Command(
        update={"board": state.board},
        goto=next_player  # All routing logic HERE
    )


async def player_one_node(state: State):
    print('player one move:')
    prompt = (PLAYER_TEMPLATE
              .replace("{{SYMBOL}}", "O")
              .replace("{{BOARD}}", str(state.board)))

    response = await Runner.run(player_one_agent, prompt)
    coord = _parse_coord(response.final_output)
    if coord is None:
        raise ValueError(f"unparsable move: {response}")
    return Command(
        update={"board": state.board, 'last_player': "O", 'moves': coord},
        goto='coordinator_node'
    )


async def player_two_node(state: State):
    print('player two move:')
    prompt = (PLAYER_TEMPLATE
              .replace("{{SYMBOL}}", "X")
              .replace("{{BOARD}}", str(state.board)))
    response: GenerateContentResponse = await player_two_agent.send_message(prompt)
    coord = _parse_coord(response.text)
    if coord is None:
        raise ValueError(f"unparsable move: {response}")
    return Command(update={"board": state.board, 'last_player': "X", 'moves': coord},
                   goto='coordinator_node')


# Build graph
builder = (
    StateGraph(State, context_schema=Context)
    .add_node("coordinator_node", coordinator_node)
    .add_node("player_one_node", player_one_node)
    .add_node("player_two_node", player_two_node)
    .add_edge(START, "coordinator_node")
    .add_edge("player_one_node", "coordinator_node")
    .add_edge("player_two_node", "coordinator_node")
)

graph = builder.compile()


# ============ ASYNC EXECUTION ============
async def run_game_async():
    """Execute graph asynchronously"""

    result = await graph.ainvoke(
        State(),
        {"configurable": {"game_id": "1"}})

    print(f"Final result: {result['game_status']}")


# ============ ASYNC STREAMING ============
async def run_game_async_streaming():
    """Stream results as they happen"""

    async for event in graph.astream({  # [!code highlight]
        "board": [['.' for _ in range(3)] for _ in range(3)],
        "moves": None,
        "last_player": None,
        "game_status": ""
    }):
        print(f"Event: {event}")


# ============ RUN ============
if __name__ == "__main__":
    # Option 1: Simple async execution
    #asyncio.run(run_game_async())

    # Option 2: With streaming
    #asyncio.run(run_game_async_streaming())

    pass
