import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# Telegram bilgilerini Railway'den al
TG_TOKEN = os.getenv("TG_TOKEN")

# /start komutu
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ Scalp AI botu çalışıyor!")

def main():
    print("🤖 Bot başlatılıyor...")

    app = ApplicationBuilder().token(TG_TOKEN).build()

    app.add_handler(CommandHandler("start", start))

    print("📡 Telegram dinleniyor...")
    app.run_polling()   # ⚠️ EN ÖNEMLİ SATIR

if __name__ == "__main__":
    main()
