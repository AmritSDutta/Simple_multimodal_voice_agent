import logging
import re

from langchain_core.messages import BaseMessage, HumanMessage
from openai.types import ModerationCreateResponse

from src.flow_agent.config import settings

# Patterns for potentially malicious content
MALICIOUS_PATTERNS = {
    # Shell command injection patterns
    "shell_injection": [
        r";\s*(rm|mv|cp|dd|chmod|chown|cat|ls)\s+",  # Command chaining with dangerous commands
        r"\|\s*(rm|mv|cp|dd|chmod|chown|cat|ls|grep)\s+",  # Pipe to dangerous commands
        r"`.*\$.*`",  # Backtick command substitution with variables
        r"\$\(.*\)",  # Command substitution
        r'&&\s*(rm|mv|dd|chmod|cat|sudo)',  # AND operator with dangerous commands
        r'\|\|\s*(rm|mv|dd|cat|sudo)',  # OR operator with dangerous commands
        r'&&\s*sudo\s+',  # AND operator with sudo
        r'\|\|\s*sudo\s+',  # OR operator with sudo
    ],
    # Docker commands that could be abused
    "docker_abuse": [
        r"docker\s+(exec|run)\s+.*-v\s*/",  # Volume mounting with root
        r"docker\s+exec.*sudo",  # Docker exec with sudo
        r"docker\s+run.*--privileged",  # Privileged mode
        r"docker\s+run.*--pid=host",  # Host PID namespace
        r"docker\s+run.*--network=host",  # Host network
        r"docker\s+exec.*chmod",  # Docker exec with chmod
        r"docker\s+exec.*sh",  # Shell access in container
        r"docker\s+exec.*bash",  # Bash access in container
    ],
    # SQL injection patterns
    "sql_injection": [
        r"(\bor\b\s*\d+\s*=\s*\d+)",  # SQL bypass with OR
        r"(\band\b\s*\d+\s*=\s*\d+)",  # SQL bypass with AND
        r"union\s+select",  # UNION SELECT injection
        r"exec\s*\(",  # Command execution in SQL
    ],
    # Path traversal attempts
    "path_traversal": [
        r"\.\./",  # Parent directory traversal
        r"\.\.\\",  # Windows parent directory
        r"%2e%2e",  # URL encoded parent directory
        r"~root",  # Root home directory access
        r"/etc/passwd",  # Accessing password file
        r"/etc/shadow",  # Accessing shadow file
        r"C:\\Windows\\System32",  # Windows system directory
    ],
    # System commands and dangerous operations
    "system_commands": [
        r"\brm\s+-rf\s+/",  # Delete root filesystem
        r"\bdd\s+if=/dev/zero",  # Disk wipe
        r"\bmkfs\.",  # Format filesystem
        r"\bshutdown\b",  # Shutdown command
        r"\breboot\b",  # Reboot command
        r"\bsystemctl\s+(stop|disable|restart)",  # System service control
        r"\bservice\s+\w+\s+stop",  # Stop services
        r"\bkill\s+-9\s+\d+",  # Kill processes
        r"\bkillall\b",  # Kill all processes
        r"\binit\s+\d",  # Change runlevel
        r"\bnc\s+.*-e",  # Netcat with execute
        r"\bbash\s+-i",  # Interactive bash
        r"\bcurl\s+.*\|\s*bash",  # Curl pipe to bash
        r"\bwget\s+.*\|\s*sh",  # Wget pipe to shell
        r"\bchmod\s+777",  # Dangerous permission change
        r"\bchmod\s+-R",  # Recursive permission change
        r"\bsudo\s+(su|bash|sh)",  # Privilege escalation
    ],
    # Script/Code execution patterns
    "code_execution": [
        r"<script[^>]*>",  # Script tags
        r"javascript:",  # JavaScript protocol
        r"onload\s*=",  # Event handler injection
        r"onerror\s*=",  # Error handler injection
        r"exec\s*\(",  # Exec function
        r"__import__\(",  # Python import injection
    ],
    # Known malicious keywords
    "malicious_keywords": [
        "malware", "virus", "trojan", "ransomware", "spyware",
        "keylogger", "backdoor", "rootkit", "botnet",
        "shellcode", "bypass",
    ],
    "windows_abuse": [
        r"\b(cmd|powershell|pwsh)\.exe\b",                 # Shell launch
        r"\bpowershell\s+-enc\b",                          # Base64 encoded PS
        r"\bpowershell\s+-nop\b",                          # NoProfile (stealth)
        r"\bwmic\s+process\s+call\s+create\b",             # Spawn process
        r"\breg\s+(add|delete|query)\b",                   # Registry tampering
        r"\bschtasks\s+/create\b",                         # Scheduled task persistence
        r"\bsc\s+(create|config|start|stop)\b",            # Service control
        r"\bbcdedit\b",                                    # Boot config tampering
        r"\bvssadmin\s+delete\s+shadows\b",                # Ransomware behavior
        r"\bcipher\s+/w\b",                                # Disk wipe
        r"\bnet\s+(user|localgroup)\b",                    # User/group manipulation
        r"\brundll32\b",                                   # LOLBIN execution
        r"\bmshta\b",                                      # Script execution LOLBIN
        r"\bcertutil\s+-decode\b",                         # Payload decode
    ]
}

# Compile all regex patterns for better performance
COMPILED_PATTERNS = {}
for category, patterns in MALICIOUS_PATTERNS.items():
    # Skip malicious_keywords - it's handled separately with context checking
    if category == "malicious_keywords":
        continue
    if isinstance(patterns, list):
        COMPILED_PATTERNS[category] = [re.compile(p, re.IGNORECASE) for p in patterns]


async def scan_for_vulnerability(user_input: HumanMessage | str) -> bool:
    """
    Scan user message for potentially malicious content.
    """
    logging.info("Scanning for vulnerabilities...")
    user_message = []
    if not user_input:
        logging.warning('empty context passed for scanning')
        return True

    if isinstance(user_input, str):
        user_message.append(user_input.lower())

    if isinstance(user_input, HumanMessage):
        content: str | list[str | dict] = user_input.content
        if isinstance(content, str):
            # Handle plain string - convert to expected format
            user_message.append(content.lower())

        if isinstance(content, list):
            for item in content:
                if hasattr(item, 'get') and item.get("type") == "text" and item.get("text"):
                    user_message.append(item.get("text").lower())

    # Convert to lowercase for keyword matching
    message_lower = str(user_message)

    # Check each category of malicious patterns
    for catgry, pttrn in COMPILED_PATTERNS.items():
        for pattern in pttrn:
            if pattern.search(message_lower):
                logging.warning(
                    f"Malicious content detected in category '{catgry}': "
                    f"pattern matched in message: {pattern} {user_message[:100]}..."
                )
                return False
    logging.info(f"Message passed keyword based vulnerability scan: {user_message[:50]}...")

    if settings.MODERATION_API_CHECK_REQ:
        moderation_check_status: bool = await get_moderation_api_feedback_on_input(user_input)
        if not moderation_check_status:
            return False

    logging.info(f"Message passed vulnerability scan: {user_message[:50]}...")
    return True


async def get_moderation_api_feedback_on_input(user_input: BaseMessage | str) -> bool:
    """
    Check input using OpenAI's moderation API for harmful content.
    (omni-moderation-latest) -> text + image
    """
    from openai import AsyncOpenAI
    if not user_input:
        logging.warning('empty context passed for moderation')
        return True

    logging.info(f"Calling OpenAI's moderation API, model: {settings.MODERATION_MODEL}")
    prompt = []

    if isinstance(user_input, str):
        prompt.append({"type": "text", "text": user_input})
    elif isinstance(user_input, HumanMessage):
        content = user_input.content
        if isinstance(content, str):
            prompt.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for item in content:
                if hasattr(item, 'get') and item.get("type") == "text":
                    prompt.append({"type": "text", "text": item.get("text", "")})
                elif hasattr(item, 'get') and item.get("type") == "image_url" and item["image_url"]:
                    prompt.append({
                        "type": "image_url",
                        "image_url": {"url": item["image_url"]["url"]}
                    })

    try:
        client = AsyncOpenAI()

        response: ModerationCreateResponse = await client.moderations.create(
            model=settings.MODERATION_MODEL,
            input=prompt,
        )

        for res in response.results:
            if res.flagged:
                logging.warning(f"Input failed moderation API check with model: {settings.MODERATION_MODEL}")
                logging.warning(f"Input flagged by moderation API. Categories: {res.categories}")
                return False

        logging.info(f"Input passed moderation API check with model: {settings.MODERATION_MODEL}")
        return True

    except Exception as e:
        logging.error(f"Moderation API error: {e}")
        return False
