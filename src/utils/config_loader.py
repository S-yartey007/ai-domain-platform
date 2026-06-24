import os
import yaml

def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Reads the config.yaml file and returns the configuration settings
    for the currently selected domain.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r") as file:
        full_config = yaml.safe_load(file)

    # Find out which domain is currently selected (e.g., 'tech_gadgets')
    selected_domain = full_config.get("selected_domain")

    if not selected_domain:
        raise ValueError("The 'selected_domain' key is missing from the configuration file.")

    # Extract only the settings for that specific domain
    domain_settings = full_config.get("domains", {}).get(selected_domain)

    if not domain_settings:
        raise ValueError(f"Domain settings for '{selected_domain}' not found in config.")

    # Inject the domain key name into the dictionary for convenience later
    domain_settings["domain_key"] = selected_domain

    return domain_settings

# Simple test block to make sure it works when run directly
if __name__ == "__main__":
    try:
        config = load_config()
        print("✅ Config loaded successfully!")
        print(f"Active Domain Name: {config.get('name')}")
        print(f"Scraper Source Type: {config.get('scraper', {}).get('source_type')}")
    except Exception as e:
        print(f"❌ Error loading config: {e}")