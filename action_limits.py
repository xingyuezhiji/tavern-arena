"""Shared turn limit for the pinned observe2 protocol (4017 features)."""

# obs2 g2 starts immediately after the 3227 observation-v1 floats.
ACTION_COUNT_FEATURE = 3227
MAX_TURN_ACTIONS = 60


def limit_action_mask(features, mask):
    """Match bot.rs: end after >60 actions, unless a choice is pending."""
    if features[ACTION_COUNT_FEATURE] > MAX_TURN_ACTIONS and mask[0]:
        mask = mask.copy()
        mask[:] = False
        mask[0] = True
    return mask
