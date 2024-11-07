from aiogram import Bot, Dispatcher, F
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.filters import Command, StateFilter
from aiogram.types import (
    Message,
    BotCommand,
    BotCommandScopeDefault,
    BotCommandScopeAllPrivateChats,
)
from aiogram.fsm.context import FSMContext
from .handlers import *
from .client import MLBotClient
from .constants import ButtonText
from .states import ModelStates
import logging

logger = logging.getLogger(__name__)


async def setup_bot(token: str, api_url: str):
    """Setup bot with handlers."""
    try:
        # Initialize bot and dispatcher
        bot = Bot(token=token)
        dp = Dispatcher(storage=MemoryStorage())
        client = MLBotClient(api_url)

        await bot.delete_my_commands(scope=BotCommandScopeDefault())
        await bot.delete_my_commands(scope=BotCommandScopeAllPrivateChats())

        await bot.set_my_commands(
            commands=[
                BotCommand(command="start", description="Start bot"),
                BotCommand(command="health", description="Check API health"),
            ],
            scope=BotCommandScopeAllPrivateChats(),
        )

        # Basic commands
        @dp.message(Command(commands=["start", "help"]))
        async def start_cmd(message: Message):
            await cmd_start(message)

        @dp.message(Command(commands=["health"]))
        @dp.message(F.text == ButtonText.HEALTH_CHECK)
        async def health_cmd(message: Message):
            await cmd_health(message, client)

        # List models
        @dp.message(F.text == ButtonText.LIST_MODELS)
        async def list_cmd(message: Message):
            await list_models(message, client)

        # Training flow
        @dp.message(F.text == ButtonText.TRAIN_MODEL)
        async def train_cmd(message: Message, state: FSMContext):
            await train_model_start(message, state)

        @dp.message(StateFilter(ModelStates.waiting_for_model_type))
        async def model_type_handler(message: Message, state: FSMContext):
            await process_model_type(message, state)

        @dp.message(
            StateFilter(ModelStates.waiting_for_training_data),
            F.content_type.in_({"document"}),
        )
        async def training_data_handler(message: Message, state: FSMContext):
            await process_training_file(message, state, client, bot)

        # Go back handler
        @dp.message(
            StateFilter(ModelStates.waiting_for_training_data),
            F.text == ButtonText.GO_BACK,
        )
        async def go_back_handler(message: Message, state: FSMContext):
            await process_go_back(message, state)

        # Prediction flow
        @dp.message(F.text == ButtonText.MAKE_PREDICTION)
        async def predict_cmd(message: Message, state: FSMContext):
            await predict_start(message, state, client)

        @dp.message(StateFilter(ModelStates.waiting_for_model_selection))
        async def model_selection_handler(message: Message, state: FSMContext):
            await process_model_selection(message, state)

        @dp.message(
            StateFilter(ModelStates.waiting_for_prediction_data),
            F.content_type.in_({"document"}),
        )
        async def prediction_data_handler(message: Message, state: FSMContext):
            await process_prediction_data(message, state, client, bot)

        # Delete model flow
        @dp.message(F.text == ButtonText.DELETE_MODEL)
        async def delete_cmd(message: Message, state: FSMContext):
            await delete_model_start(message, state, client)

        @dp.message(StateFilter(ModelStates.waiting_for_delete_selection))
        async def delete_selection_handler(message: Message, state: FSMContext):
            await process_delete_selection(message, state, client)

        # Error handlers
        @dp.error()
        async def error_handler(event: types.ErrorEvent):
            logger.error(f"Update error: {event.exception}", exc_info=True)
            try:
                # Try to notify user about error
                if event.update.message:
                    await event.update.message.answer(
                        "An error occurred while processing your request.\n"
                        "Please try again or contact support.",
                        reply_markup=get_main_keyboard(),
                    )
            except Exception as e:
                logger.error(f"Error in error handler: {e}")

        # Register middleware if needed
        # dp.middleware.setup(YourMiddleware())

        logger.info("Bot setup completed successfully")
        return dp, bot

    except Exception as e:
        logger.error(f"Failed to setup bot: {e}")
        raise
