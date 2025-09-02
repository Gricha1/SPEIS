import sys
from tests.enjoy_validation import register_custom_components, custom_parse_args
from tests.validation import enjoy

def main():
    """Script entry point."""
    register_custom_components()
    cfg = custom_parse_args(evaluation=True)
    # print(f"config {cfg}")
    status = enjoy(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())