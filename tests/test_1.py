import pytest

from main import cmd_rate

from aiogram_tests import MockedBot
from aiogram_tests.handler import MessageHandler
from aiogram_tests.types.dataset import MESSAGE


@pytest.mark.asyncio
async def test_echo():
    request = MockedBot(MessageHandler(cmd_rate))
    calls = await request.query()
    answer_message = calls.send_messsage.fetchone()
    assert answer_message.text == "Hello, Bot!"
