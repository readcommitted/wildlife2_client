import logging
import traceback


def custom_exception_hook(exc_type, exc_value, tb):
    full_traceback = "".join(traceback.format_exception(exc_type, exc_value, tb))
    first_line = f"{exc_type.__name__}: {exc_value}"
    short_message = f"{exc_type.__name__}"

    # Log full traceback silently
    logging.error(full_traceback)

    # Also log just the first line for quick visibility
    logging.error(f"First line: {first_line}")

    return short_message

