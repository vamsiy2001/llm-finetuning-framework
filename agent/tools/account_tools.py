"""
Mock account management tools for the customer support agent.
"""

import random
from datetime import datetime, timedelta
from typing import Optional


def get_account_info(customer_email: str) -> dict:
    """
    Retrieve customer account information.

    Args:
        customer_email: The customer's email address

    Returns:
        dict with account details or error
    """
    if "@" not in customer_email:
        return {"error": f"Invalid email address: {customer_email}"}

    random.seed(hash(customer_email))
    plans = ["Basic", "Standard", "Premium"]
    statuses = ["active", "active", "active", "suspended", "cancelled"]

    created_days_ago = random.randint(30, 1000)
    created_date = datetime.now() - timedelta(days=created_days_ago)

    return {
        "email": customer_email,
        "account_status": random.choice(statuses),
        "subscription_plan": random.choice(plans),
        "member_since": created_date.strftime("%Y-%m-%d"),
        "total_orders": random.randint(1, 50),
        "loyalty_points": random.randint(0, 5000),
        "payment_method": "Visa ending in 4242",
        "billing_cycle": "monthly",
        "next_billing_date": (datetime.now() + timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d"),
    }


def unlock_account(customer_email: str) -> dict:
    """
    Unlock a customer account that was locked due to failed login attempts.

    Args:
        customer_email: The customer's email address

    Returns:
        dict with unlock status
    """
    account = get_account_info(customer_email)
    if "error" in account:
        return account

    if account["account_status"] == "cancelled":
        return {
            "success": False,
            "error": "Cannot unlock a cancelled account. Please create a new account.",
        }

    return {
        "success": True,
        "email": customer_email,
        "message": (
            "Your account has been unlocked. "
            "A password reset link has been sent to your email. "
            "The link expires in 1 hour."
        ),
        "action_taken": "account_unlocked + password_reset_sent",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
    }


def update_subscription(customer_email: str, new_plan: str) -> dict:
    """
    Change a customer's subscription plan.

    Args:
        customer_email: The customer's email address
        new_plan: The new plan name ("Basic", "Standard", "Premium")

    Returns:
        dict with subscription change details
    """
    valid_plans = {"Basic": 9.99, "Standard": 19.99, "Premium": 39.99}

    if new_plan not in valid_plans:
        return {
            "error": f"Invalid plan: {new_plan}. Available plans: {list(valid_plans.keys())}"
        }

    account = get_account_info(customer_email)
    if "error" in account:
        return account

    if account["account_status"] in ("suspended", "cancelled"):
        return {
            "success": False,
            "error": f"Cannot change plan — account is {account['account_status']}.",
        }

    old_plan = account["subscription_plan"]
    if old_plan == new_plan:
        return {
            "success": False,
            "error": f"Account is already on the {new_plan} plan.",
        }

    old_price = valid_plans[old_plan]
    new_price = valid_plans[new_plan]
    is_upgrade = new_price > old_price

    return {
        "success": True,
        "email": customer_email,
        "old_plan": old_plan,
        "new_plan": new_plan,
        "old_price": f"${old_price:.2f}/month",
        "new_price": f"${new_price:.2f}/month",
        "effective_date": "immediately" if is_upgrade else "next billing cycle",
        "proration_credit": f"${abs(new_price - old_price):.2f}" if is_upgrade else "N/A",
        "message": (
            f"{'Upgraded' if is_upgrade else 'Downgraded'} from {old_plan} to {new_plan}. "
            f"{'New features are available now.' if is_upgrade else 'Current plan active until next billing date.'}"
        ),
    }


def cancel_subscription(customer_email: str, reason: Optional[str] = None) -> dict:
    """
    Cancel a customer's subscription.

    Args:
        customer_email: The customer's email address
        reason: Optional cancellation reason

    Returns:
        dict with cancellation details
    """
    account = get_account_info(customer_email)
    if "error" in account:
        return account

    if account["account_status"] == "cancelled":
        return {
            "success": False,
            "error": "Subscription is already cancelled.",
        }

    return {
        "success": True,
        "email": customer_email,
        "cancellation_date": datetime.now().strftime("%Y-%m-%d"),
        "access_until": account["next_billing_date"],
        "reason": reason,
        "message": (
            f"Subscription cancelled. You'll retain access until {account['next_billing_date']}. "
            "No further charges will be made."
        ),
        "reactivation_info": "You can reactivate your subscription at any time.",
    }


def process_refund(customer_email: str, order_id: str, amount: float, reason: str) -> dict:
    """
    Process a refund for a customer.

    Args:
        customer_email: The customer's email address
        order_id: The order ID to refund
        amount: Refund amount in dollars
        reason: Reason for the refund

    Returns:
        dict with refund confirmation
    """
    if amount <= 0:
        return {"error": "Refund amount must be greater than zero."}

    if amount > 500:
        return {
            "success": False,
            "error": f"Refund of ${amount:.2f} exceeds the automated limit ($500). "
                     "A senior agent will review and process this within 24 hours.",
            "escalated": True,
        }

    account = get_account_info(customer_email)
    if "error" in account:
        return account

    return {
        "success": True,
        "refund_id": f"REF-{hash(order_id + customer_email) % 1000000:06d}",
        "order_id": order_id,
        "email": customer_email,
        "amount": f"${amount:.2f}",
        "method": account["payment_method"],
        "reason": reason,
        "eta": "3-5 business days",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
        "message": f"Refund of ${amount:.2f} initiated to your {account['payment_method']}.",
    }
