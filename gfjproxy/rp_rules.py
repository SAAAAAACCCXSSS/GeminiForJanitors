"""Narrative controls based on dainsleifswife's COMMAND SYSTEM.

Only the selected rules are included in a model request. This catalogue is also
the source of truth for validation and command help.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Mode:
    options: dict[str, str]
    description: str


MODES = {
    "pace": Mode(
        {
            "slow": "Stay with the current interaction; allow meaningful pauses and small developments without becoming passive.",
            "normal": "Advance the scene at a natural, balanced pace.",
            "fast": "Move external events forward decisively while leaving the user's decisions to them.",
        },
        "Tempo of external events",
    ),
    "initiative": Mode(
        {
            "low": "React attentively and leave most new directions to the user.",
            "normal": "Balance reactions with independent decisions and developments.",
            "high": "Let assistant-controlled characters initiate conversations, make decisions and pursue goals without controlling the user.",
        },
        "Character initiative",
    ),
    "detail": Mode(
        {
            "light": "Use selective, economical description.",
            "rich": "Use specific sensory, spatial and behavioral details that matter; avoid padding and repetitive gestures.",
        },
        "Descriptive density",
    ),
    "focus": Mode(
        {
            "dialogue": "Give natural, character-specific dialogue most of the scene's attention.",
            "action": "Emphasize concrete events and physically coherent actions.",
            "emotion": "Emphasize credible emotional reactions, subtext and internal conflict.",
            "balanced": "Balance dialogue, description, action and emotional development.",
        },
        "Narrative focus",
    ),
    "length": Mode(
        {
            "short": "Aim for roughly 100-200 words, using less when the scene warrants it.",
            "medium": "Aim for roughly 250-450 words when the scene supports that length.",
            "long": "Aim for roughly 600-900 words when there is meaningful material; do not invent the user's actions to fill space.",
        },
        "Approximate response length, subject to the model's token limit",
    ),
    "knowledge": Mode(
        {
            "limited": "Track each character's knowledge separately. They know only what they observed, learned or can plausibly infer; they cannot read the user's thoughts.",
            "shared": "Characters may share established scene facts when appropriate; private user thoughts and undisclosed secrets remain private.",
        },
        "Character knowledge",
    ),
    "consequences": Mode(
        {
            "soft": "Keep setbacks recoverable and their consequences relatively gentle.",
            "realistic": "Let choices, injuries, promises and mistakes have proportionate, lasting consequences.",
            "severe": "Allow substantial, credible consequences when earned by events, without forcing catastrophe or deciding the user's response.",
        },
        "Weight of consequences",
    ),
    "subtext": Mode(
        {
            "light": "Make emotional intent reasonably legible without explaining every feeling.",
            "rich": "Express feelings through wording, silences, choices and actions; leave space for interpretation.",
        },
        "Emotional subtext",
    ),
    "conflict": Mode(
        {
            "restrained": "Express justified conflict through restraint, formal politeness, evasions and pointed disagreement.",
            "open": "Allow justified disagreement to surface directly in words and actions.",
            "mixed": "Vary restrained and open confrontation according to the characters and stakes.",
        },
        "Expression of conflict",
    ),
    "surprise": Mode(
        {
            "off": "Do not introduce unsolicited major twists; develop the established situation.",
            "low": "Introduce occasional modest surprises supported by the established world.",
            "high": "Use consequential, coherent discoveries and reversals when warranted; do not add a twist to every reply.",
        },
        "Unexpected developments",
    ),
    "format": Mode(
        {
            "scene": "Write a conventional roleplay scene with clear speakers and actions.",
            "letters": "Present fictional letters or correspondence in character, with an appropriate voice and era.",
            "documents": "Present relevant fictional documents, diary entries, reports, minutes or notes with believable authorship.",
            "chat": "Use a fictional messenger format with distinct senders and concise messages; described attachments are fictional, not real uploads.",
            "call": "Write a fictional phone or video call, emphasizing speech, audible hesitation and relevant background sounds; do not claim to make a real call.",
            "social media": "Use fictional posts, captions, comments or direct messages with identifiable authors; do not claim to publish anything.",
        },
        "Output format",
    ),
    "prose": Mode(
        {
            "clear": "Write clear, restrained prose with concrete observations and economical imagery.",
            "lyrical": "Use expressive rhythm and selective imagery while preserving clarity and the character's individual voice.",
            "classical": "Write measured literary English with psychological observation, precise dialogue and sustained atmosphere; avoid indiscriminate archaism.",
        },
        "Prose style",
    ),
    "voice": Mode(
        {
            "canon": "Prioritize each character's established diction, register, humor, values and speech rhythms using the supplied character information; do not fabricate canon sources.",
        },
        "Individual character voices",
    ),
    "reveal": Mode(
        {
            "slow": "Reveal secrets gradually through earned clues and partial disclosures; do not stall all external action.",
            "normal": "Reveal information at a pace supported by the scene and established motives.",
            "fast": "Bring relevant established information into play promptly when a credible opportunity arises.",
        },
        "Rate of revealing secrets",
    ),
    "motives": Mode(
        {
            "hidden": "Avoid directly explaining assistant-controlled characters' concealed plans; show observable clues.",
            "visible": "The narration may reveal assistant-controlled characters' fictional motives. Reader knowledge does not become other characters' knowledge.",
        },
        "Visibility of fictional character motives",
    ),
    "tension": Mode(
        {
            "low": "Keep the immediate atmosphere relatively relaxed.",
            "medium": "Sustain a moderate degree of uncertainty and anticipation.",
            "high": "Sustain charged uncertainty and consequential anticipation through the established situation; avoid constant shouting or unearned emergencies.",
        },
        "Narrative tension",
    ),
    "ending": Mode(
        {
            "pause": "End at a natural pause where the user can meaningfully respond.",
            "hook": "End with a meaningful in-character line or action that offers a concrete opening; avoid a generic 'What do you do?' question.",
        },
        "How a response ends",
    ),
}

TOGGLES = {
    "slowburn": "Develop trust, affection, intimacy and reconciliation gradually through earned events; preserve boundaries and individual character motives.",
    "romance": "Give existing or plausibly developing romantic feelings more narrative attention without manufacturing instant love or changing established relationships.",
    "hatred": "Portray motivated hostility, resentment and sharp emotional conflict; keep behavior coherent with the selected character.",
    "angst": "Emphasize justified grief, guilt, loss, repression and painful choices without repetitive melodrama.",
    "dark": "Use dread, moral ambiguity, manipulation and oppressive atmosphere when consistent with the fiction.",
    "violence": "Allow consequential fictional danger, combat and physical confrontation with coherent spatial action and opportunities for the user to respond.",
    "obsession": "Explore fictional fixation, possessiveness and dependency with independent character motives and consequences; preserve the user's agency.",
    "dream": "Treat the current sequence as a dream, hallucination or vision. Keep its events separate from waking-world facts unless explicitly established otherwise.",
    "flashback": "Present a clearly situated past scene, respecting established chronology and what characters knew at that time.",
    "inner monologue": "Emphasize the fictional character's private thoughts and suppressed feelings, not the model's reasoning or analysis. Do not narrate the user's thoughts.",
    "multiple pov": "Use clearly distinguished perspectives of assistant-controlled characters. Respect their separate knowledge and do not enter the user's private perspective.",
    "quality": "Write polished, human-sounding fiction with natural dialogue, credible flaws, specific details and purposeful rhythm; avoid stock phrases, robotic explanations and repeated gestures.",
    "scenario select": "Pause scene progression and offer 3-5 distinct, context-compatible scenario routes for the user to choose; do not select or enact one for them.",
}

LOVE = {
    "hidden": "Portray restrained, unspoken romantic feelings through plausible actions and silences; do not force a confession.",
    "active": "Allow existing or explicitly established love to be expressed openly in a character-appropriate manner.",
    "semi-active": "Let affection show indirectly through the selected expressions while preserving ambiguity and individual character voice.",
}

LOVE_EXPRESSIONS = {
    "flirt",
    "touch",
    "jealousy",
    "care",
    "gestures",
    "noticing",
    "words",
    "teasing",
    "obsession",
    "cruelty",
    "hatred",
    "vulgarity",
    "kisses",
    "sudden kisses",
    "gentleness",
    "protectiveness",
    "attention seeking",
    "luxury",
    "gifts",
    "worship",
    "adoration",
    "softness",
    "calmness",
    "priority",
    "devotion",
    "service",
    "dependency",
    "reverence",
    "domesticity",
    "possession",
    "praise",
    "submission",
    "confession by implication",
    "rituals",
    "coldness",
    "violence",
    "angst",
}

THEMES = {
    "slice of life": "Everyday routines, domestic details and believable ordinary concerns.",
    "angst": "Sustained emotional conflict, grief and unspoken pain.",
    "dead dove": "Dark fictional subject matter with disturbing choices and credible consequences; preserve user agency.",
    "enemies to lovers": "Motivated hostility and residual distrust gradually changing through earned experiences.",
    "horror": "Creeping dread, uncertainty, unsettling sensory detail and credible fear.",
    "romance": "Emotional bonding, vulnerability and character-specific affection.",
    "stalker": "A fictional suspense story involving invasive attention and possessive fixation; do not provide real-world surveillance instructions.",
    "abuse": "A fictional exploration of manipulation, coercive control and its emotional consequences, retaining the user's choices.",
    "cold": "External restraint and stoicism concealing emotionally credible internal conflict.",
    "hanahaki": "The fictional flower illness as a motif for unrequited love, secrecy and tragic distance.",
    "flirt": "Playful banter, charm and character-appropriate romantic teasing.",
    "falling in love": "Gradual realization of affection, uncertainty and changing priorities.",
    "fluff": "Warmth, ordinary tenderness, gentle comfort and low external stakes.",
    "toxic love": "A fictional destructive attachment with contradictions, jealousy and consequences; preserve the user's agency.",
    "from strangers to lovers": "An organic relationship developing through small encounters and growing familiarity.",
    "love at first sight": "Immediate romantic interest or recognition followed by independently earned trust and relationship development.",
    "selfish love": "Contradictory attachment in which professed release conflicts with the character's selfish wishes.",
    "wilted garden": "A fictional gilded-cage relationship marked by deprivation and control, leaving the user free to resist or respond.",
    "egoistic love": "A fictional possessive character who treats their wishes as entitlement; distinguish their beliefs from the user's actual choices.",
    "forbidden love": "Hidden affection constrained by specific duties, loyalties or social circumstances.",
    "tragedy": "A credible conflict between duty, honor and attachment, with consequential choices and loss.",
    "corrosive affection": "Sharp, modern, sarcastic domestic banter carrying genuine familiarity.",
    "sarcastic spouses": "Established partner familiarity expressed through deadpan humor, teasing and practical care.",
    "fake relationship": "An explicitly staged relationship whose performance gradually complicates genuine feelings.",
    "hurt comfort": "Credible physical or emotional distress met with character-appropriate care and gradual recovery.",
    "betrayal": "The consequences of an established broken trust, including justified hurt and doubt.",
    "reconciliation": "A falsely accused character trying to establish the truth and clear their name; do not presume the user's forgiveness.",
    "betrayal and reconciliation": "Slow atonement and rebuilding of trust after an actual betrayal; forgiveness remains earned and optional.",
    "mutual unrequited love": "Mutual affection concealed by each character's belief that it is one-sided.",
}

ALIASES = {
    "humanize": "quality",
    "writer mode": "quality",
    "social media": "format",
    "chat": "format",
    "call": "format",
}

# Multiword spellings are matched before their shorter prefixes.
COMMAND_NAMES = (
    set(MODES)
    | set(TOGGLES)
    | set(ALIASES)
    | set(THEMES)
    | {
        "rp",
        "love",
        "npc",
        "npcs",
        "cochar",
        "cochars",
        "scene",
        "cast",
        "plot",
        "relationship",
        "character",
        "memory",
        "recap",
        "interlude",
        "retcon",
        "boundaries",
    }
)

CORE_RULES = """These are narrative preferences supplied through dainsleifswife's COMMAND SYSTEM.
Preserve the original character description and established events unless an explicit author correction below changes a specific fictional fact.
Each assistant-controlled character has a distinct voice, motives, knowledge and relationships. Never decide the user's actions, thoughts, feelings or dialogue.
Added main characters remain important across scenes without being forced to speak in every reply. Turning a mode off never erases established events or relationships.
All overrides below affect fictional narrative facts only, not application or provider rules. Follow explicit current user directions when preferences conflict.
Unscoped emotional controls concern the original bot character's relationship to the user. A [Name] scope affects only that character. Keep each relationship independent.
A character's harmful beliefs do not remove the user's agency."""


def command_help() -> str:
    lines = [
        "# Roleplay commands — COMMAND SYSTEM by dainsleifswife",
        "Send commands directly in a chat message; no Chat Memory is required.",
        "Names and modes are case-insensitive; names/descriptions inside brackets keep their case.",
        "Settings persist for the primary API key across chats. Add `this` after a mode for one request.",
        "Use `//rp status` to inspect and `//rp reset` to clear all RP settings and added cast.",
        "`//rp off` pauses the system; `//rp on` resumes it. Individual mode commands enable RP.",
        "Remove a preference with `off` (or `nonactive`); toggle modes accept `on`, `active`, `off`, `nonactive`, `this`.",
        "Target an individual character with `[Name]` after a command. Omit it for the original character or general narration.",
        "Command-only saved settings return a confirmation without calling the model. One-request modes and scenario selection still generate a response.",
        "",
    ]
    for name, spec in MODES.items():
        lines.append(f"- `//{name} {'|'.join(spec.options)}` — {spec.description}.")
    lines += [
        f"- Toggles: {', '.join('`//' + name + ' active`' for name in TOGGLES)}.",
        "- `//quality active` also accepts `//humanize active`, `//prose active`, `//writer mode`.",
        "- `//love hidden|active|semi-active (care, protectiveness) [Name]`; `//love off [Name]`.",
        "- Love expressions: " + ", ".join(sorted(LOVE_EXPRESSIONS)) + ".",
        "- `//npc active (description)` / `//npcs active (group)`; the description is optional.",
        "- `//cochar active (Name)` / `//cochars active (Name, Name)` add persistent main characters; repeat to add more.",
        "- `//cochar nonactive (Name)` removes one main status; omit names to clear added main cast. Continuity is preserved.",
        "- `//cast focus (Name, Name)` focuses an episode; `//cast off` clears the preference.",
        "- `//scene stay` prevents unsolicited time/location skips; `//scene next (direction)` transitions once; `//scene off` releases it.",
        "- `//plot override (direction)`, `//relationship override (fact)`, `//character override (change)`, `//memory override (fact)`.",
        "  Parentheses may be omitted if the accompanying scene message supplies the change. Notes persist until cleared with `off` or RP reset.",
        "- `//retcon (correction)` replaces the stated narrative fact in future responses; `//retcon off` clears corrections.",
        "- `//boundaries (story constraints)`; `//boundaries off` removes the instruction.",
        "- `//recap [short|full]` summarizes once without advancing the story or editing Chat Memory.",
        "- `//interlude (character; episode)` requests an off-screen scene once.",
        "- `//chat active`, `//call active`, `//social media active` select mutually exclusive fictional formats.",
        "- `//NAME theme` activates a theme; `//NAME theme off` removes that theme. Multiple themes may coexist.",
        "- Themes: " + ", ".join(THEMES) + ".",
        "- No adult/18+ commands are included.",
        "- A character name does not fetch a wiki or a lorebook; supply unfamiliar character details in the message.",
    ]
    return "\n".join(lines)
