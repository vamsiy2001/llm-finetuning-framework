"""
Mock order management tools for the customer support agent.
In production these would call your actual order management API.
"""

import random
from datetime import datetime, timedelta
from typing import Optional


def lookup_order(order_id: str) -> dict:
    """
    Look up order status by order ID.

    Args:
        order_id: The order ID (e.g. "ORD-123456")

    Returns:
        dict with order details or an error if not found.
    """
    if not order_id.startswith("ORD-"):
        return {"error": f"Invalid order ID format: {order_id}. Expected format: ORD-XXXXXX"}

    # simulate realistic order states
    random.seed(hash(order_id))
    statuses = ["processing", "shipped", "out_for_delivery", "delivered", "cancelled"]
    carriers = ["FedEx", "UPS", "USPS", "DHL"]
    status = random.choice(statuses)
    carrier = random.choice(carriers)

    order_date = datetime.now() - timedelta(days=random.randint(1, 14))
    estimated_delivery = order_date + timedelta(days=random.randint(3, 10))

    result = {
        "order_id": order_id,
        "status": status,
        "order_date": order_date.strftime("%Y-%m-%d"),
        "estimated_delivery": estimated_delivery.strftime("%Y-%m-%d"),
        "carrier": carrier,
        "tracking_number": f"{carrier[:2].upper()}{random.randint(100000000, 999999999)}",
        "items": [
            {"name": "Product A", "quantity": 1, "price": 49.99},
            {"name": "Product B", "quantity": 2, "price": 19.99},
        ],
        "total": 89.97,
        "shipping_address": "123 Main St, Springfield, IL 62701",
    }

    if status == "delivered":
        result["delivered_date"] = (estimated_delivery - timedelta(days=1)).strftime("%Y-%m-%d")

    return result


def update_shipping_address(order_id: str, new_address: str) -> dict:
    """
    Update the shipping address for an unshipped order.

    Args:
        order_id: The order ID
        new_address: The new shipping address

    Returns:
        dict with success/failure status
    """
    order = lookup_order(order_id)
    if "error" in order:
        return order

    if order["status"] in ("shipped", "out_for_delivery", "delivered"):
        return {
            "success": False,
            "error": f"Cannot update address — order is already {order['status']}. "
                     "Contact the carrier directly to redirect the package.",
            "carrier": order["carrier"],
            "tracking_number": order["tracking_number"],
        }

    return {
        "success": True,
        "order_id": order_id,
        "old_address": order["shipping_address"],
        "new_address": new_address,
        "message": "Shipping address updated successfully.",
    }


def cancel_order(order_id: str, reason: Optional[str] = None) -> dict:
    """
    Cancel an order if it hasn't shipped yet.

    Args:
        order_id: The order ID
        reason: Optional cancellation reason

    Returns:
        dict with success/failure and refund info
    """
    order = lookup_order(order_id)
    if "error" in order:
        return order

    if order["status"] in ("shipped", "out_for_delivery", "delivered"):
        return {
            "success": False,
            "error": f"Cannot cancel — order is already {order['status']}. "
                     "Please use our return process once the package arrives.",
        }

    if order["status"] == "cancelled":
        return {
            "success": False,
            "error": "Order is already cancelled.",
        }

    return {
        "success": True,
        "order_id": order_id,
        "cancelled_at": datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
        "refund_amount": order["total"],
        "refund_method": "original payment method",
        "refund_eta": "3-5 business days",
        "reason": reason,
    }


def get_return_label(order_id: str, item_name: str, return_reason: str) -> dict:
    """
    Generate a return shipping label for a delivered order.

    Args:
        order_id: The order ID
        item_name: Name of the item to return
        return_reason: Reason for the return

    Returns:
        dict with return label info
    """
    order = lookup_order(order_id)
    if "error" in order:
        return order

    if order["status"] != "delivered":
        return {
            "success": False,
            "error": f"Returns are only available for delivered orders. "
                     f"Current status: {order['status']}.",
        }

    delivered_date = order.get("delivered_date", order["estimated_delivery"])
    delivered_dt = datetime.strptime(delivered_date, "%Y-%m-%d")
    days_since_delivery = (datetime.now() - delivered_dt).days

    if days_since_delivery > 30:
        return {
            "success": False,
            "error": f"Return window has expired. Items must be returned within 30 days of delivery "
                     f"(delivered {days_since_delivery} days ago).",
        }

    return {
        "success": True,
        "order_id": order_id,
        "item": item_name,
        "return_reason": return_reason,
        "return_label_url": f"https://returns.example.com/label/{order_id}",
        "return_instructions": (
            "1. Pack the item securely in original packaging if possible.\n"
            "2. Print and attach the return label.\n"
            "3. Drop off at any authorized carrier location.\n"
            "4. Refund processed within 5-7 days of receiving your return."
        ),
        "refund_eta": "5-7 business days after receipt",
        "carrier": "USPS",
    }
