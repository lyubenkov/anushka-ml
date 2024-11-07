from aiogram import types
from aiogram.fsm.context import FSMContext
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from .states import ModelStates
from .keyboards import get_main_keyboard, get_model_types_keyboard
from .utils import process_data_file, format_model_info, format_prediction_result
from .client import MLBotClient
from .constants import ButtonText
import logging
from io import BytesIO
import pandas as pd

logger = logging.getLogger(__name__)


async def cmd_start(message: Message):
    """Handle /start command."""
    await message.answer(
        "Welcome to Anushka ML Bot! 🤖\n"
        "I can help you train and manage ML models.\n\n"
        "Available commands:\n"
        f"{ButtonText.HEALTH_CHECK} - Check API status\n"
        f"{ButtonText.TRAIN_MODEL} - Train a new model\n"
        f"{ButtonText.MAKE_PREDICTION} - Make predictions\n"
        f"{ButtonText.LIST_MODELS} - List available models\n"
        f"{ButtonText.DELETE_MODEL} - Delete a model\n\n"
        "Use the menu buttons below or commands from the menu button (hamburger icon)",
        reply_markup=get_main_keyboard(),
    )


async def cmd_health(message: Message, client: MLBotClient):
    """Handle health check button."""
    try:
        health_status = await client.health_check()
        await message.answer(
            f"API Status: {health_status['status']}\n"
            f"Timestamp: {health_status['timestamp']}\n\n"
            "✅ API is responding normally",
            reply_markup=get_main_keyboard(),
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        await message.answer("❌ API is not responding")


async def train_model_start(message: Message, state: FSMContext):
    """Start model training process."""
    await state.set_state(ModelStates.waiting_for_model_type)
    await message.answer("Choose model type:", reply_markup=get_model_types_keyboard())


async def process_model_type(message: Message, state: FSMContext):
    """Process selected model type."""
    await state.update_data(model_type=message.text)
    await state.set_state(ModelStates.waiting_for_training_data)

    # Create keyboard with "Go Back" button
    keyboard = ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=ButtonText.GO_BACK)]],
        resize_keyboard=True,
        one_time_keyboard=True,
    )

    await message.answer(
        "Please upload your training data file (CSV or Excel format).\n"
        "The file should contain features and a 'label' column.",
        reply_markup=keyboard,
    )


async def process_training_file(
    message: Message, state: FSMContext, client: MLBotClient, bot
):
    """Process uploaded training file and train model."""
    try:
        if not message.document:
            await message.answer("Please upload a file")
            return

        file = await bot.get_file(message.document.file_id)
        file_content = await bot.download_file(file.file_path)

        try:
            status_message = await message.answer(
                "Processing your file...\n" "☑️ File received\n" "  Reading data..."
            )

            X, y = process_data_file(file_content.read(), is_training=True)

            await status_message.edit_text(
                "Processing your file...\n"
                "☑️ File received\n"
                "☑️ Data validated\n"
                f"Training model with samples..."
            )

            state_data = await state.get_data()
            model_type = state_data["model_type"]

            result = await client.train_model(
                features=X, labels=y, model_type=model_type
            )

            # Create a shorter response message
            model_id = result.get("model_id", "N/A")
            metrics = result.get("metrics", {})

            # Format metrics concisely
            metrics_text = []
            for k, v in metrics.items():
                if isinstance(v, float):
                    metrics_text.append(f"{k}: {v:.3f}")
                else:
                    metrics_text.append(f"{k}: {v}")

            response = f"✅ Model trained successfully!\n" f"Model ID: {model_id}\n"

            if metrics_text:
                response += f"Metrics:\n" + "\n".join(metrics_text[:3])

            await message.answer(response, reply_markup=get_main_keyboard())

        except pd.errors.EmptyDataError:
            await message.answer(
                "The uploaded file is empty", reply_markup=get_main_keyboard()
            )
        except ValueError as ve:
            await message.answer(
                f"Error processing file: {str(ve)[:200]}",  # Truncate error message
                reply_markup=get_main_keyboard(),
            )

    except Exception as e:
        logger.error(f"Training failed: {e}")
        # Truncate error message if it's too long
        error_msg = str(e)[:200] + "..." if len(str(e)) > 200 else str(e)
        await message.answer(
            f"Training failed: {error_msg}", reply_markup=get_main_keyboard()
        )
    finally:
        await state.clear()


async def list_models(message: types.Message, client: MLBotClient):
    """List available models."""
    try:
        models = await client.get_available_models()

        available_models_text = "\nAvailable Model Types:\n"
        for model_type in models["available_models"]:
            pretty_name = model_type.replace("_", " ").title()
            available_models_text += f"  * {pretty_name}\n"

        active_models_text = ""
        if models["active_models"]:
            active_models_text = "🟢 Active Models:\n"
            for model_id, model_type in models["active_models"].items():
                active_models_text += f"  * ID: {model_id}\n    Type: {model_type}\n"
        else:
            active_models_text = "🔴 No active models\n"

        models_text = f"{active_models_text}\n" f"{available_models_text}\n"

        await message.answer(text=models_text, reply_markup=get_main_keyboard())
    except Exception as e:
        logger.error(f"Failed to get models: {e}")
        await message.answer(
            text="Failed to get models list", reply_markup=get_main_keyboard()
        )


async def predict_start(message: types.Message, state: FSMContext, client: MLBotClient):
    """Start prediction process."""
    try:
        models = await client.get_available_models()
        if not models["active_models"]:
            await message.answer(
                "No models available for prediction.", reply_markup=get_main_keyboard()
            )
            await state.clear()
            return

        # Store models in state
        await state.update_data(available_models=models["active_models"])
        await state.set_state(ModelStates.waiting_for_model_selection.state)

        models_text = "\n".join(
            f"{i+1}. {model_id} ({model_type})"
            for i, (model_id, model_type) in enumerate(models["active_models"].items())
        )

        await message.answer(
            f"Select model for prediction:\n\n{models_text}\n\n"
            "Enter the number of the model you want to use.",
            reply_markup=get_main_keyboard(),
        )
    except Exception as e:
        logger.error(f"Failed to get models: {e}")
        await message.answer(
            "Failed to start prediction process", reply_markup=get_main_keyboard()
        )
        await state.clear()


async def process_model_selection(message: types.Message, state: FSMContext):
    """Process model selection for prediction."""
    try:
        state_data = await state.get_data()
        models = state_data["available_models"]
        model_index = int(message.text) - 1

        if 0 <= model_index < len(models):
            model_id = list(models.keys())[model_index]
            await state.update_data(selected_model_id=model_id)
            await state.set_state(ModelStates.waiting_for_prediction_data)
            await message.answer(
                "Please upload prediction data file (CSV or Excel)\n"
                "File should contain features only"
            )
        else:
            await message.answer("Invalid selection. Please try again.")
    except (ValueError, IndexError):
        await message.answer("Invalid input. Please enter a valid number.")


async def process_prediction_data(
    message: types.Message, state: FSMContext, client: MLBotClient, bot
):
    """Process prediction data and make predictions."""
    try:
        file = await bot.get_file(message.document.file_id)
        file_content = await bot.download_file(file.file_path)

        X, _ = process_data_file(file_content.read(), is_training=False)
        state_data = await state.get_data()
        model_id = state_data["selected_model_id"]

        result = await client.predict(features=X, model_id=model_id)

        await message.answer(format_prediction_result(result))
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        await message.answer(f"Prediction failed: {str(e)}")
    finally:
        await state.clear()


async def delete_model_start(
    message: types.Message, state: FSMContext, client: MLBotClient
):
    """Start model deletion process."""
    try:
        models = await client.get_available_models()
        if not models["active_models"]:
            await message.answer(
                "No models available for deletion.", reply_markup=get_main_keyboard()
            )
            return

        await state.update_data(available_models=models["active_models"])
        await state.set_state(ModelStates.waiting_for_delete_selection)

        models_text = "\n".join(
            f"{i+1}. {model_id} ({model_type})"
            for i, (model_id, model_type) in enumerate(models["active_models"].items())
        )
        await message.answer(
            f"Select model to delete:\n\n{models_text}\n\n"
            "Enter the number of the model you want to delete.",
            reply_markup=get_main_keyboard(),
        )
    except Exception as e:
        logger.error(f"Failed to get models: {e}")
        await message.answer(
            "Failed to start deletion process", reply_markup=get_main_keyboard()
        )
        await state.clear()


async def process_delete_selection(
    message: types.Message, state: FSMContext, client: MLBotClient
):
    """Process model deletion selection."""
    try:
        state_data = await state.get_data()
        models = state_data["available_models"]
        try:
            model_index = int(message.text) - 1

            if 0 <= model_index < len(models):
                model_id = list(models.keys())[model_index]
                try:
                    await client.delete_model(model_id)
                    await message.answer(
                        f"✅ Model {model_id} has been deleted successfully.",
                        reply_markup=get_main_keyboard(),
                    )
                except Exception as e:
                    logger.error(f"Failed to delete model: {e}")
                    await message.answer(
                        f"❌ Failed to delete model: {str(e)}",
                        reply_markup=get_main_keyboard(),
                    )
            else:
                await message.answer(
                    "Invalid selection. Please try again or use /start to return to main menu.",
                    reply_markup=get_main_keyboard(),
                )
        except ValueError:
            await message.answer(
                "Please enter a valid number or use /start to return to main menu.",
                reply_markup=get_main_keyboard(),
            )
    finally:
        await state.clear()


async def process_go_back(message: Message, state: FSMContext):
    """Handle go back button press."""
    await state.set_state(ModelStates.waiting_for_model_type)
    await message.answer(
        "Choose a model type:", reply_markup=get_model_types_keyboard()
    )
