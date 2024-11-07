from aiogram.fsm.state import State, StatesGroup


class ModelStates(StatesGroup):
    """States for model training and prediction workflow."""

    waiting_for_model_type = State()
    waiting_for_training_data = State()
    waiting_for_model_selection = State()
    waiting_for_prediction_data = State()
    waiting_for_delete_selection = State()
