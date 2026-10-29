"""User-confirmed product groups for the 17 source Shopee listing IDs.

These names were explicitly supplied by the project owner. Grouping affects
display aggregation only: original listing IDs, review IDs and rows remain
unchanged. Unknown IDs are never assigned to one of these groups by inference.
"""

PRODUCT_GROUPS: dict[str, tuple[str, ...]] = {
    "Baseus H17 Encok 3.5mm Wired Earphone": ("14750761025", "27956652952"),
    "Baseus C17 Type-C Earphones": ("29224862918", "19495504570", "15004032602", "27104868593"),
    "Baseus Encok CZ11 / HZ11": ("28305736055", "28007706814", "29005735910"),
    "Baseus WM01 TWS Wireless Earphones": ("27020823629", "11315577801", "4039048325"),
    "Baseus H19": ("25502231352", "28656652968", "22050094510", "23987350200", "7678007496"),
}

PRODUCT_NAMES: dict[str, str] = {
    listing_id: product_name
    for product_name, listing_ids in PRODUCT_GROUPS.items()
    for listing_id in listing_ids
}
