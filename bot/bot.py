import logging
import os, psutil, threading, asyncio

process = psutil.Process(os.getpid())

from io import BytesIO
from aiogram import Bot, Dispatcher, types
from aiogram.utils.executor import start_webhook
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.contrib.fsm_storage.files import JSONStorage

from .settings import (BOT_TOKEN, HEROKU_APP_NAME,
                          WEBHOOK_URL, WEBHOOK_PATH,
                          WEBAPP_HOST, WEBAPP_PORT,
                       WEBHOOK_SSL_CERT, WEBHOOK_SSL_PRIV)

from .style_transfer import image_loader, getMask, run_style_transfer, getImage, mse_loss


logging.basicConfig(level=logging.INFO)
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot, storage=JSONStorage("file.json"))
dp.middleware.setup(LoggingMiddleware())


@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    await message.reply("Hi!\nI'm Style bot!\nSend me an image with caption 'original' and an image with caption 'style'.\nThen send command /execute.")


@dp.message_handler(content_types=[types.message.ContentType.PHOTO, types.message.ContentType.DOCUMENT])
async def images_handler(message: types.Message):
    if message.caption in ['style', 'original'] and message and message.document:
        d = await dp.storage.get_data(chat=message.chat.id, user=message.from_user.id)
        d[message.caption] = message.document.file_id
        await dp.storage.set_data(chat=message.chat.id, user=message.from_user.id, data=d)
        dp.storage.write(dp.storage.path)


@dp.message_handler(commands=['execute'])
async def execute_handler(message: types.Message):
    mem("execute start")

    v = await dp.storage.get_data(chat=message.chat.id, user=message.from_user.id)

    original_file_id = v['original'] if 'original' in v else None
    style_file_id = v['style'] if 'original' in v else None

    if not original_file_id:
        await message.reply("Please send original image.")
        return

    if not style_file_id:
        await message.reply("Please send style image.")
        return

    original_file = await bot.download_file_by_id(original_file_id)
    style_file = await bot.download_file_by_id(style_file_id)

    original_file = image_loader(original_file)
    style_file = image_loader(style_file)

    mask = getMask()

    content_img = original_file
    style1_img = style_file
    style2_img = style_file

    input_img = content_img.clone()

    await message.answer("Style transfer in process. It takes about 10 minutes on heroku...")

    t = threading.Thread(
        target=lambda message, content_img, style1_img, style2_img, input_img, mask:
        asyncio.run(process_nst(message, content_img, style1_img, style2_img, input_img, mask)),
        args=(message, content_img, style1_img, style2_img, input_img, mask))

    t.start()


async def process_nst(message, content_img, style1_img, style2_img, input_img, mask):
    mem("run_style_transfer start")

    output_mse = run_style_transfer(content_img, style1_img, style2_img, input_img, mask, mask,
                                    style2_weight=0, content_weight=7, loss_fn=mse_loss, num_steps=200,
                                    timer=show_progress(message, 200), memory_notifier=mem)

    output_img = getImage(output_mse)

    answer_image = BytesIO()
    answer_image.name = 'image.jpeg'
    output_img.save(answer_image, 'JPEG')
    answer_image.seek(0)

    bot1 = Bot(token=BOT_TOKEN)

    await bot1.send_photo(message.chat.id, photo=answer_image)

    await bot1.close_bot()


@dp.message_handler()
async def echo(message: types.Message):
    print(message.from_user.username)
    await message.answer(message.text)


def show_progress(message, steps):
    async def show_progress_int(i):
        await message.answer(f"{i*100//steps} %")
    return show_progress_int


async def on_startup(dp):
    logging.warning('Starting connection.')
    logging.warning(f"PORT: {WEBAPP_PORT}")
    mem("before webhook")
    await bot.set_webhook(WEBHOOK_URL, allowed_updates=["message"], drop_pending_updates=True)#, certificate=open(WEBHOOK_SSL_CERT, 'rb'))
    mem("after webhook")


async def on_shutdown(dp):
    logging.warning('Bye! Shutting down webhook connection')
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.storage.close()


def mem(txt = None):
    m = process.memory_info().rss // 2 ** 20
    logging.warning(f"used memory: {m} Mb - {txt}")

def main():
    logging.basicConfig(level=logging.INFO)
    start_webhook(
        dispatcher=dp,
        webhook_path=WEBHOOK_PATH,
        skip_updates=True,
        on_startup=on_startup,
        host=WEBAPP_HOST,
        port=WEBAPP_PORT,
    )


if __name__ == "__main__":
    main()