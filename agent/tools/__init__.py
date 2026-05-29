from agent.tools.account_tools import (
    cancel_subscription,
    get_account_info,
    process_refund,
    unlock_account,
    update_subscription,
)
from agent.tools.order_tools import (
    cancel_order,
    get_return_label,
    lookup_order,
    update_shipping_address,
)

ALL_TOOLS = [
    lookup_order,
    update_shipping_address,
    cancel_order,
    get_return_label,
    get_account_info,
    unlock_account,
    update_subscription,
    cancel_subscription,
    process_refund,
]
