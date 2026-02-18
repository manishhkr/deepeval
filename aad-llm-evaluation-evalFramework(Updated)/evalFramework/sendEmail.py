import os
import smtplib
from pathlib import Path
from datetime import datetime
from email.message import EmailMessage
from email.utils import formatdate

from dotenv import load_dotenv


def load_email_cfg(env_file: str = ".env") -> dict:
    load_dotenv(env_file)

    cfg = {
        "SMTP_SERVER": os.getenv("SMTP_SERVER", ""),
        "SMTP_PORT": int(os.getenv("SMTP_PORT", "25")),
        "SMTP_FROM": os.getenv("SMTP_FROM", ""),
        "EMAIL_TO": os.getenv("EMAIL_TO", ""),
        "EMAIL_SUBJECT_PREFIX": os.getenv("EMAIL_SUBJECT_PREFIX", "AI Evaluation Report"),
    }

    missing = [k for k, v in cfg.items() if v in ("", None) and k != "SMTP_PORT"]
    if missing:
        raise ValueError(f"Missing required email env vars in {env_file}: {', '.join(missing)}")

    return cfg


def get_today_report_path(out_base_dir: str = "output", report_filename: str = "report_offline.html") -> Path:
    today = datetime.now().strftime("%Y-%m-%d")
    return Path(out_base_dir) / today / report_filename


def send_report_email(
    subject_suffix: str = "REPORT",
    body_lines: list[str] | None = None,
    out_base_dir: str = "output",
    env_file: str = ".env",
) -> None:
    """
    Always sends an email and attaches today's offline report:
      output/YYYY-MM-DD/report_offline.html
    """
    cfg = load_email_cfg(env_file)
    report_path = get_today_report_path(out_base_dir=out_base_dir)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    subject = f"{cfg['EMAIL_SUBJECT_PREFIX']} - {subject_suffix} - {ts}"

    if body_lines is None:
        body_lines = [
            "AI Evaluation Framework - Report Notification",
            "",
            f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Report    : {report_path}",
            "",
            "Attachment: report_offline.html (offline HTML report)",
        ]

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg["SMTP_FROM"]
    msg["To"] = cfg["EMAIL_TO"]
    msg["Date"] = formatdate(localtime=True)
    msg.set_content("\n".join(body_lines))

    # Attach report (if missing, still send the email)
    if report_path.exists():
        msg.add_attachment(
            report_path.read_bytes(),
            maintype="text",
            subtype="html",
            filename=report_path.name,
        )
    else:
        # Still send email, but mention missing attachment
        body_lines.append("")
        body_lines.append(f"⚠ Report not found, attachment skipped: {report_path}")
        msg.set_content("\n".join(body_lines))

    with smtplib.SMTP(cfg["SMTP_SERVER"], int(cfg["SMTP_PORT"])) as server:
        server.send_message(msg)

    print("✉ Email sent (report attachment attempted).")
