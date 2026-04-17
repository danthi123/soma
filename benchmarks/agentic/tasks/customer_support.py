"""Customer support with ticket history (40-80 turns).

The agent is a support rep who must resolve a customer issue by
referencing 5 prior tickets loaded at setup.  Tests whether SOMA's
retrieval surfaces the right past tickets when the customer describes
a related problem after 30+ turns of troubleshooting.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

from benchmarks.agentic.metrics import TaskResult

# ---------------------------------------------------------------------------
# Ticket databases (one per scenario variant)
# ---------------------------------------------------------------------------
_SCENARIOS: list[dict] = [
    # Scenario 0: billing + authentication overlap
    {
        "tickets": [
            {
                "id": "TK-1001",
                "subject": "Cannot log in after password reset",
                "customer": "alice@example.com",
                "status": "resolved",
                "created": "2025-12-01",
                "body": (
                    "Customer reset password via email link. New "
                    "password accepted at reset page but login still "
                    "fails with 'invalid credentials'. Root cause: "
                    "password hash was not propagated to the auth "
                    "cache. Fix: flush auth cache after password reset."
                ),
                "resolution": "Flushed auth cache; customer confirmed login works.",
                "tags": ["auth", "password", "cache"],
            },
            {
                "id": "TK-1002",
                "subject": "Double charge on monthly subscription",
                "customer": "bob@example.com",
                "status": "resolved",
                "created": "2025-12-05",
                "body": (
                    "Customer was charged twice for the Pro plan in "
                    "December. Investigation: payment webhook fired "
                    "twice due to timeout retry. Refund issued for "
                    "duplicate charge."
                ),
                "resolution": "Refund of $29.99 issued; webhook idempotency check added.",
                "tags": ["billing", "subscription", "refund"],
            },
            {
                "id": "TK-1003",
                "subject": "API rate limit too aggressive",
                "customer": "carol@example.com",
                "status": "resolved",
                "created": "2025-12-10",
                "body": (
                    "Customer on Enterprise plan hitting 429 errors "
                    "after 50 req/min. Enterprise limit should be "
                    "500 req/min. Root cause: plan tier was not "
                    "synced to rate limiter after upgrade."
                ),
                "resolution": "Synced plan tier to rate limiter; customer confirmed.",
                "tags": ["api", "rate-limit", "enterprise"],
            },
            {
                "id": "TK-1004",
                "subject": "SSO login fails with SAML error",
                "customer": "dave@example.com",
                "status": "resolved",
                "created": "2025-12-15",
                "body": (
                    "Enterprise customer cannot log in via SSO. SAML "
                    "response shows clock skew > 5 minutes. Customer's "
                    "IdP server clock was 7 minutes ahead. Temporary "
                    "fix: increased skew tolerance to 10 min. "
                    "Permanent fix: customer synced NTP."
                ),
                "resolution": "Customer synced NTP; SSO working. Skew tolerance reverted.",
                "tags": ["auth", "sso", "enterprise"],
            },
            {
                "id": "TK-1005",
                "subject": "Subscription downgrade not reflected in billing",
                "customer": "eve@example.com",
                "status": "resolved",
                "created": "2025-12-20",
                "body": (
                    "Customer downgraded from Pro to Basic but was "
                    "still charged Pro rate. Root cause: billing "
                    "system uses cached plan tier that refreshes "
                    "daily. Downgrade happened at 11pm, billing ran "
                    "at 11:30pm before cache refresh."
                ),
                "resolution": "Refund of price difference; cache now refreshes on plan change.",
                "tags": ["billing", "subscription", "cache"],
            },
        ],
        "new_issue": (
            "Hi, I'm frank@example.com. I just upgraded from Basic "
            "to Enterprise yesterday, but I'm having two problems: "
            "(1) I still can't access the Enterprise API endpoints "
            "and I'm getting rate limited at 50 req/min, and "
            "(2) my SSO configuration isn't working -- I get a "
            "SAML error when trying to log in through our company "
            "IdP. Can you help?"
        ),
        "expected_ticket_refs": ["TK-1003", "TK-1004", "TK-1005"],
        "resolution_keywords": [
            "sync",
            "rate limit",
            "plan tier",
            "cache",
            "sso",
            "saml",
            "clock",
            "ntp",
        ],
    },
    # Scenario 1: data export + permissions overlap
    {
        "tickets": [
            {
                "id": "TK-2001",
                "subject": "CSV export missing columns",
                "customer": "grace@example.com",
                "status": "resolved",
                "created": "2026-01-05",
                "body": (
                    "Customer exports data as CSV but 'created_at' "
                    "and 'updated_at' columns are missing. Root "
                    "cause: timestamp columns excluded by default "
                    "serializer. Fix: added timestamps to default "
                    "export schema."
                ),
                "resolution": "Added timestamp columns to default export schema.",
                "tags": ["export", "csv", "schema"],
            },
            {
                "id": "TK-2002",
                "subject": "Team member cannot access shared dashboard",
                "customer": "henry@example.com",
                "status": "resolved",
                "created": "2026-01-10",
                "body": (
                    "New team member added to workspace but cannot "
                    "see shared dashboards. Root cause: dashboard "
                    "permissions use role-based access and new "
                    "members default to 'viewer' role which excludes "
                    "shared dashboards. Fix: updated default role to "
                    "include shared dashboard read access."
                ),
                "resolution": "Updated default role permissions for shared dashboards.",
                "tags": ["permissions", "dashboard", "roles"],
            },
            {
                "id": "TK-2003",
                "subject": "Scheduled report emails not sending",
                "customer": "ivy@example.com",
                "status": "resolved",
                "created": "2026-01-15",
                "body": (
                    "Customer set up daily report emails but they "
                    "stopped arriving 3 days ago. Root cause: email "
                    "service rate limit. Customer has 50 recipients "
                    "but limit is 25/batch. Fix: added pagination "
                    "for email batches."
                ),
                "resolution": "Added email batch pagination; reports now send to all recipients.",
                "tags": ["email", "reports", "rate-limit"],
            },
            {
                "id": "TK-2004",
                "subject": "Data export times out for large datasets",
                "customer": "jack@example.com",
                "status": "resolved",
                "created": "2026-01-20",
                "body": (
                    "Export of 100K+ rows times out with 504 error. "
                    "Root cause: synchronous export blocks the "
                    "request thread. Fix: moved large exports to "
                    "async job queue with download link emailed "
                    "when ready."
                ),
                "resolution": "Moved large exports to async queue; email notification added.",
                "tags": ["export", "performance", "async"],
            },
            {
                "id": "TK-2005",
                "subject": "Webhook deliveries failing silently",
                "customer": "karen@example.com",
                "status": "resolved",
                "created": "2026-01-25",
                "body": (
                    "Customer's webhook endpoint returns 200 but "
                    "payload is empty. Root cause: webhook serializer "
                    "strips fields the customer's role doesn't have "
                    "permission to access. Customer had 'viewer' role "
                    "on the webhook config. Fix: webhook uses system "
                    "role for serialization."
                ),
                "resolution": "Changed webhook serializer to use system role; full payload sent.",
                "tags": ["webhook", "permissions", "serialization"],
            },
        ],
        "new_issue": (
            "Hi, I'm leo@example.com. I added 5 new team members "
            "to our workspace last week. Now I'm seeing three issues: "
            "(1) They can't see any of the shared dashboards I set up, "
            "(2) our scheduled CSV export is missing the timestamp "
            "columns even though I configured them, and "
            "(3) webhooks to our internal system are sending empty "
            "payloads for events triggered by the new members. "
            "Can you investigate?"
        ),
        "expected_ticket_refs": ["TK-2001", "TK-2002", "TK-2005"],
        "resolution_keywords": [
            "permission",
            "role",
            "viewer",
            "dashboard",
            "schema",
            "timestamp",
            "webhook",
            "serializ",
        ],
    },
]


# ---------------------------------------------------------------------------
# Task implementation
# ---------------------------------------------------------------------------
@dataclass
class CustomerSupportTask:
    """Customer support benchmark with ticket history."""

    seed: int = 0
    _step: int = 0
    _tickets: list[dict] = field(default_factory=list)
    _scenario: dict = field(default_factory=dict)
    _referenced_tickets: set[str] = field(default_factory=set)
    _resolution_submitted: bool = False
    _resolution_text: str = ""
    _escalated: bool = False
    _customer_replies: list[str] = field(default_factory=list)
    _done: bool = False
    _max_steps: int = 80
    _tool_errors: int = 0
    _rng: random.Random = field(default_factory=random.Random)

    TOOLS: list[dict] = field(
        default_factory=lambda: [
            {
                "type": "function",
                "function": {
                    "name": "search_tickets",
                    "description": (
                        "Search historical support tickets by query."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_ticket",
                    "description": "Get full details of a ticket by ID.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Ticket ID (e.g. TK-1001)",
                            },
                        },
                        "required": ["id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "reply_to_customer",
                    "description": "Send a reply to the customer.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Reply message",
                            },
                        },
                        "required": ["message"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "escalate",
                    "description": "Escalate the ticket to a senior agent.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Reason for escalation",
                            },
                        },
                        "required": ["reason"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "resolve",
                    "description": (
                        "Mark the ticket as resolved with a solution."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "solution": {
                                "type": "string",
                                "description": "Resolution summary",
                            },
                        },
                        "required": ["solution"],
                    },
                },
            },
        ]
    )

    def setup(self) -> str:
        self._rng = random.Random(self.seed)
        idx = self.seed % len(_SCENARIOS)
        self._scenario = _SCENARIOS[idx]
        self._tickets = self._scenario["tickets"]
        self._referenced_tickets = set()
        self._resolution_submitted = False
        self._resolution_text = ""
        self._escalated = False
        self._customer_replies = []
        self._step = 0
        self._done = False
        self._tool_errors = 0

        # Build initial observation with ticket summaries
        ticket_summaries = []
        for t in self._tickets:
            ticket_summaries.append(
                f"  [{t['id']}] {t['subject']} "
                f"({t['status']}, {t['created']})"
            )
        tickets_str = "\n".join(ticket_summaries)

        return (
            "You are a customer support agent. You have access to "
            "the ticket history system.\n\n"
            f"Historical tickets in the system:\n{tickets_str}\n\n"
            "A new customer issue has come in:\n\n"
            f"{self._scenario['new_issue']}\n\n"
            "Use search_tickets and get_ticket to find relevant "
            "prior cases, then resolve the issue."
        )

    def execute_action(self, action: str) -> str:
        self._step += 1
        if self._step >= self._max_steps:
            self._done = True
            return "Maximum steps reached."

        tool_name, args = _parse_tool_call(action)

        if tool_name == "search_tickets":
            return self._handle_search(args)
        if tool_name == "get_ticket":
            return self._handle_get_ticket(args)
        if tool_name == "reply_to_customer":
            return self._handle_reply(args)
        if tool_name == "escalate":
            return self._handle_escalate(args)
        if tool_name == "resolve":
            return self._handle_resolve(args)

        self._tool_errors += 1
        return (
            f"Error: Unknown tool '{tool_name}'. Available: "
            "search_tickets, get_ticket, reply_to_customer, "
            "escalate, resolve"
        )

    def _handle_search(self, args: dict) -> str:
        query = args.get("query", "").lower()
        matches: list[dict[str, str]] = []
        for t in self._tickets:
            searchable = (
                f"{t['subject']} {t['body']} "
                f"{' '.join(t['tags'])}"
            ).lower()
            # Simple keyword matching
            query_words = query.split()
            if any(w in searchable for w in query_words):
                matches.append({
                    "id": t["id"],
                    "subject": t["subject"],
                    "status": t["status"],
                    "tags": t["tags"],
                })
        if not matches:
            return json.dumps({"results": [], "total": 0})
        return json.dumps({"results": matches, "total": len(matches)})

    def _handle_get_ticket(self, args: dict) -> str:
        ticket_id = args.get("id", "")
        for t in self._tickets:
            if t["id"] == ticket_id:
                self._referenced_tickets.add(ticket_id)
                return json.dumps(t)
        return json.dumps({"error": f"Ticket '{ticket_id}' not found."})

    def _handle_reply(self, args: dict) -> str:
        message = args.get("message", "")
        self._customer_replies.append(message)
        # Track ticket references in replies
        for t in self._tickets:
            if t["id"] in message:
                self._referenced_tickets.add(t["id"])
        # Simulate customer acknowledgment
        return (
            "Customer response: Thank you for looking into this. "
            "Please let me know what you find."
        )

    def _handle_escalate(self, args: dict) -> str:
        self._escalated = True
        self._done = True
        return "Ticket escalated to senior support team."

    def _handle_resolve(self, args: dict) -> str:
        solution = args.get("solution", "")
        self._resolution_submitted = True
        self._resolution_text = solution
        self._done = True
        # Track ticket references in resolution
        for t in self._tickets:
            if t["id"] in solution:
                self._referenced_tickets.add(t["id"])
        return "Ticket resolved. Customer has been notified."

    def is_complete(self) -> bool:
        return self._done or self._step >= self._max_steps

    def score(self) -> TaskResult:
        expected_refs = set(self._scenario["expected_ticket_refs"])
        correct_refs = self._referenced_tickets & expected_refs
        ref_score = (
            len(correct_refs) / len(expected_refs) * 100
            if expected_refs
            else 0.0
        )

        # Resolution quality: check for keywords
        resolution_score = 0.0
        if self._resolution_submitted:
            keywords = self._scenario["resolution_keywords"]
            all_text = (
                self._resolution_text
                + " "
                + " ".join(self._customer_replies)
            ).lower()
            matched = sum(
                1 for kw in keywords if kw.lower() in all_text
            )
            resolution_score = (
                matched / len(keywords) * 100 if keywords else 0.0
            )

        # Combined accuracy: 60% ticket refs + 40% resolution quality
        accuracy = ref_score * 0.6 + resolution_score * 0.4

        return TaskResult(
            completion=self._resolution_submitted
            and len(correct_refs) >= len(expected_refs) // 2,
            accuracy=accuracy,
            steps=self._step,
            tool_errors=self._tool_errors,
            extra={
                "referenced_tickets": sorted(self._referenced_tickets),
                "expected_tickets": sorted(expected_refs),
                "correct_refs": len(correct_refs),
                "total_expected_refs": len(expected_refs),
                "resolution_submitted": self._resolution_submitted,
                "escalated": self._escalated,
                "customer_replies": len(self._customer_replies),
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_tool_call(action: str) -> tuple[str, dict]:
    """Parse a tool call from the agent's response."""
    try:
        parsed = json.loads(action)
        if isinstance(parsed, dict):
            name = parsed.get("tool") or parsed.get("name", "")
            args = parsed.get("arguments") or parsed.get("args", {})
            if isinstance(args, str):
                args = json.loads(args)
            return name, args
    except (json.JSONDecodeError, TypeError):
        pass
    return "", {}
