def xtract_device_family(user_agent_str):
    if isinstance(user_agent_str, str):
        user_agent_lower = user_agent_str.lower()
        if "mobile" in user_agent_lower or "android" in user_agent_lower or "iphone" in user_agent_lower or "ipad" in user_agent_lower:
            return "Tablet" if "ipad" in user_agent_lower else "Mobile"
        elif "tablet" in user_agent_lower or "ipad" in user_agent_lower:
            return "Tablet"
        elif "windows" in user_agent_lower or "macintosh" in user_agent_lower or "linux" in user_agent_lower:
            return "Desktop"
        else:
            return "Other"
    return "Unknown"

def xtract_os_family(user_agent_str):
    if isinstance(user_agent_str, str):
        user_agent_lower = user_agent_str.lower()
        if "windows" in user_agent_lower:
            return "Windows"
        elif "macintosh" in user_agent_lower or "macos" in user_agent_lower:
            return "MacOS"
        elif "android" in user_agent_lower:
            return "Android"
        elif "ios" in user_agent_lower or "iphone" in user_agent_lower or "ipad" in user_agent_lower:
            return "iOS"
        elif "linux" in user_agent_lower:
            return "Linux"
        else:
            return "Other"
    return "Unknown"

def xtract_browser_family(user_agent_str):
    if isinstance(user_agent_str, str):
        user_agent_lower = user_agent_str.lower()
        if "chrome" in user_agent_lower:
            return "Chrome"
        elif "firefox" in user_agent_lower:
            return "Firefox"
        elif "safari" in user_agent_lower and "chrome" not in user_agent_lower:
            return "Safari"
        elif "edge" in user_agent_lower:
            return "Edge"
        elif "msie" in user_agent_lower or "trident" in user_agent_lower:
            return "IE"
        else:
            return "Other"
    return "Unknown"
