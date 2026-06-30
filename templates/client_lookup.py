# templates/client_lookup.py

def build_client_spoken(client: dict) -> str:
    full_name = f"{client['first_name']} {client['last_name']}"
    return (
        f"Found {full_name}. "
        f"Phone: {client['phone']}. "
        f"Address: {client['address']}."
    )


def build_client_list_spoken(clients: list[dict]) -> str:
    if not clients:
        return "No clients found matching that name."
    if len(clients) == 1:
        return build_client_spoken(clients[0])
    names = ", ".join(f"{c['first_name']} {c['last_name']}" for c in clients)
    return f"I found {len(clients)} clients: {names}. Which one did you mean?"