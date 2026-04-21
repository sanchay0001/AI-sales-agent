def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Mock API function to capture a qualified lead.
    Only called after all three fields are collected.
    """
    print(f"\n🎯 Lead captured successfully: {name}, {email}, {platform}\n")
    return f"Lead captured successfully: {name}, {email}, {platform}"