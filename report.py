from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
import json

console = Console(legacy_windows=False)


def _score_color(score: float) -> str:
    if score >= 8:
        return "green"
    if score >= 5:
        return "yellow"
    return "red"


def _bool_icon(val: bool) -> str:
    return "✅" if val else "❌"


def print_report(analysis: dict, audio_file: str = ""):
    console.print()
    console.rule("[bold blue]📞 ניתוח שיחה - רשת אצה[/bold blue]")
    if audio_file:
        console.print(f"[dim]קובץ: {audio_file}[/dim]\n")

    # --- הזמנה ---
    order = analysis["order"]
    order_text = Text()
    order_text.append("פריטים שהוזמנו:\n", style="bold")
    for item in order["items"]:
        order_text.append(f"  • {item}\n")
    order_text.append(f"\nכתובת: {order['address']}\n")
    if order.get("special_requests"):
        order_text.append(f"בקשות מיוחדות: {order['special_requests']}\n")
    order_text.append(f"\nנציג חזר על הזמנה: {_bool_icon(order['repeated_back_to_customer'])}  ")
    order_text.append(f"הזמנה אושרה: {_bool_icon(order['confirmed_by_agent'])}")
    console.print(Panel(order_text, title="[bold cyan]🛒 פרטי ההזמנה[/bold cyan]", border_style="cyan"))

    # --- ביצועי נציג ---
    agent = analysis["agent_performance"]
    score = agent["overall_score"]
    color = _score_color(score)

    script_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    script_table.add_column("תחנה", style="white")
    script_table.add_column("בוצע", justify="center")
    for checkpoint, done in agent["script_compliance"].items():
        script_table.add_row(checkpoint, _bool_icon(done))

    console.print(Panel(
        script_table,
        title=f"[bold {color}]👤 ביצועי נציג | ציון: {score}/10[/bold {color}]",
        border_style=color,
    ))

    if agent["missed_checkpoints"]:
        console.print(f"  [red]תחנות שהוחסרו:[/red] {', '.join(agent['missed_checkpoints'])}")
    if agent["improvement_areas"]:
        console.print(f"  [yellow]לשיפור:[/yellow] {', '.join(agent['improvement_areas'])}")
    console.print()

    # --- שביעות רצון לקוח ---
    cust = analysis["customer_satisfaction"]
    cust_score = cust["overall_score"]
    cust_color = _score_color(cust_score)
    sentiment_map = {"חיובי": "🟢", "שלילי": "🔴", "נייטרלי": "🟡"}
    icon = sentiment_map.get(cust["sentiment"], "⚪")

    cust_text = Text()
    cust_text.append(f"סנטימנט כללי: {icon} {cust['sentiment']}\n\n", style="bold")
    if cust["frustration_indicators"]:
        cust_text.append("סימני תסכול:\n", style="red bold")
        for f in cust["frustration_indicators"]:
            cust_text.append(f"  • {f}\n", style="red")
    if cust["satisfaction_indicators"]:
        cust_text.append("סימני שביעות רצון:\n", style="green bold")
        for s in cust["satisfaction_indicators"]:
            cust_text.append(f"  • {s}\n", style="green")
    if cust.get("notes"):
        cust_text.append(f"\n{cust['notes']}")

    console.print(Panel(
        cust_text,
        title=f"[bold {cust_color}]😊 שביעות רצון לקוח | ציון: {cust_score}/10[/bold {cust_color}]",
        border_style=cust_color,
    ))

    # --- ניתוח מחלוקות ---
    dispute = analysis["dispute_analysis"]
    dispute_text = Text()
    dispute_text.append("מה הלקוח ביקש:\n", style="bold")
    for item in dispute["order_stated_by_customer"]:
        dispute_text.append(f"  • {item}\n")
    if dispute["order_corrections"]:
        dispute_text.append("\nתיקונים במהלך השיחה:\n", style="yellow bold")
        for c in dispute["order_corrections"]:
            dispute_text.append(f"  • {c}\n", style="yellow")
    dispute_text.append(f"\nנציג אימת הזמנה: {_bool_icon(dispute['agent_verified_order'])}\n\n")
    dispute_text.append("קביעת אחריות:\n", style="bold")
    dispute_text.append(dispute["liability_assessment"])

    console.print(Panel(dispute_text, title="[bold magenta]⚖️ ניתוח מחלוקות הזמנה[/bold magenta]", border_style="magenta"))

    # --- איכות שיחה ---
    call_q = analysis["call_quality"]
    duration_min = int(call_q["duration_seconds"]) // 60
    duration_sec = int(call_q["duration_seconds"]) % 60
    console.print(Panel(
        f"משך: {duration_min}:{duration_sec:02d}  |  קצב: {call_q['pace_assessment']}  |  "
        f"מילים/דקה: {call_q['words_per_minute']}  |  בהירות: {call_q['clarity_score']}/10",
        title="[bold white]📊 נתוני שיחה[/bold white]",
        border_style="white",
    ))

    # --- סיכום ---
    console.print(Panel(
        f"[italic]{analysis['summary']}[/italic]",
        title="[bold blue]📝 סיכום[/bold blue]",
        border_style="blue",
    ))
    console.print()
