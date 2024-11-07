from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
from .constants import ButtonText


def get_main_keyboard() -> ReplyKeyboardMarkup:
    """Get main menu keyboard."""
    keyboard = [
        [
            KeyboardButton(text=ButtonText.HEALTH_CHECK),
        ],
        [
            KeyboardButton(text=ButtonText.LIST_MODELS),
            KeyboardButton(text=ButtonText.TRAIN_MODEL),
        ],
        [
            KeyboardButton(text=ButtonText.MAKE_PREDICTION),
            KeyboardButton(text=ButtonText.DELETE_MODEL),
        ],
    ]
    return ReplyKeyboardMarkup(keyboard=keyboard, resize_keyboard=True)


def get_model_types_keyboard() -> ReplyKeyboardMarkup:
    """Get model types selection keyboard."""
    keyboard = [
        [
            KeyboardButton(text="random_forest"),
            KeyboardButton(text="svm"),
        ],
        # [
        #     KeyboardButton(text="linear_regression"),
        #     KeyboardButton(text="gradient_boosting"),
        # ],
    ]
    return ReplyKeyboardMarkup(
        keyboard=keyboard, resize_keyboard=True, one_time_keyboard=True
    )


def remove_keyboard() -> ReplyKeyboardRemove:
    """Remove keyboard."""
    return ReplyKeyboardRemove()
