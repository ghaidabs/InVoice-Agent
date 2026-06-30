from mock.db import (
    search_clients_by_name, get_client_by_id,
    search_products_by_name, get_product_by_id,
    get_invoice_by_number, get_invoices_by_client_name, get_invoices_by_status,
    create_invoice, add_line_item,
    build_invoice_draft, confirm_invoice_draft,
)


class MockSaasClient:
    def __init__(self, company_id: int, user_id: int):
        self.company_id = company_id
        self.user_id = user_id


    async def lookup_client(self, name: str) -> list[dict]:
        return search_clients_by_name(name, self.company_id)

    async def get_client(self, client_id: int) -> dict | None:
        return get_client_by_id(client_id, self.company_id)


    async def lookup_product(self, name: str) -> list[dict]:
        return search_products_by_name(name, self.company_id)

    async def get_product(self, product_id: int) -> dict | None:
        return get_product_by_id(product_id, self.company_id)


    async def get_invoice(self, invoice_number: str) -> dict | None:
        return get_invoice_by_number(invoice_number, self.company_id)

    async def get_invoices_by_client(self, client_name: str) -> list[dict]:
        return get_invoices_by_client_name(client_name, self.company_id)

    async def get_invoices_by_status(self, status: str) -> list[dict]:
        return get_invoices_by_status(status, self.company_id)

    async def create_invoice(
        self,
        client: dict,
        line_items: list[dict],
        currency: str = "TND",
        is_tva_applicable: bool = False,
        is_stamp_applicable: bool = False,
        payment_method: str | None = None,
        terms_and_conditions: str | None = None,
        due_date: str | None = None,
        ) -> dict:
            return create_invoice(
                company_id=self.company_id,
                user_id=self.user_id,
                client=client,
                line_items=line_items,
                currency=currency,
                is_tva_applicable=is_tva_applicable,
                is_stamp_applicable=is_stamp_applicable,
                payment_method=payment_method,
                terms_and_conditions=terms_and_conditions,
                due_date=due_date,
            )

    async def build_invoice_draft(self, client: dict, line_items: list, **kwargs) -> dict:
        return build_invoice_draft(
            company_id=self.company_id,
            user_id=self.user_id,
            client=client,
            line_items=line_items,
            **kwargs,
        )

    async def confirm_invoice_draft(self, draft: dict) -> dict:
        return confirm_invoice_draft(draft)

        
    async def add_line_item(
        self,
        invoice_number: str,
        designation: str,
        quantity: float,
        unit_price: float,
        unit: str,
        description: str | None = None,
    ) -> dict | None:
        return add_line_item(
            invoice_number=invoice_number,
            company_id=self.company_id,
            designation=designation,
            quantity=quantity,
            unit_price=unit_price,
            unit=unit,
            description=description,
        )