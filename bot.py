import os
import json
import logging
import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import re
import random
from enum import Enum
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)
import pytz
from dotenv import load_dotenv

# ==================== НАСТРОЙКА ====================
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ==================== ENUMS ====================
class AutoReplyMode(Enum):
    OFF = "off"
    WORK_HOURS = "work_hours"
    ALWAYS = "always"
    CUSTOM = "custom"
    VACATION = "vacation"
    SICK = "sick"

class UserStatus(Enum):
    AVAILABLE = "available"
    BUSY = "busy"
    MEETING = "meeting"
    VACATION = "vacation"
    SICK = "sick"
    LUNCH = "lunch"

# ==================== DEEPSEEK ИИ ====================
class DeepSeekAI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1"
        self.model = "deepseek-chat"

    async def chat(self, messages: List[Dict], max_tokens: int = 300) -> Optional[str]:
        """Отправляет запрос к DeepSeek API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "stream": False
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        logger.error(f"DeepSeek API error: {response.status} - {error_text}")
                        return None
        except Exception as e:
            logger.error(f"DeepSeek connection error: {e}")
            return None

# ==================== КЛАСС АВТООТВЕТЧИКА ====================
class AutoReplyManager:
    def __init__(self):
        self.responses = {
            "default": "👩‍💼 Я сейчас занята. Отвечу вам в ближайшее время!",
            "work_hours": "👩‍💼 Рабочий день окончен. Отвечу завтра с 9:00.",
            "busy": "👩‍💼 В данный момент я занята. Перезвоню вам позже.",
            "meeting": "👩‍💼 Я на совещании. Отвечу после его окончания.",
            "vacation": "👩‍💼 Я в отпуске до {date}. По срочным вопросам напишите 'СРОЧНО'.",
            "sick": "👩‍💼 Я болею. Вернусь к работе {date}. Спасибо за понимание.",
            "lunch": "👩‍💼 Я на обеденном перерыве. Вернусь в {time}.",
            "weekend": "👩‍💼 Сегодня выходной. Отвечу в понедельник.",
            "night": "👩‍💼 Сейчас нерабочее время. Отвечу утром."
        }

    def get_response(self, mode: AutoReplyMode, status: UserStatus,
                    params: Dict = None, message: str = None) -> Optional[str]:
        """Получает ответ автоответчика"""
        if mode == AutoReplyMode.OFF:
            return None
        params = params or {}
        # Пользовательский автоответ
        if mode == AutoReplyMode.CUSTOM and params.get("custom_message"):
            return params["custom_message"]
        # Статус "В отпуске"
        if status == UserStatus.VACATION:
            date = params.get("vacation_end", "неизвестно когда")
            return self.responses["vacation"].format(date=date)
        # Статус "Болею"
        if status == UserStatus.SICK:
            date = params.get("sick_until", "скоро")
            return self.responses["sick"].format(date=date)
        # Статус "Занята"
        if status == UserStatus.BUSY:
            return self.responses["busy"]
        # Статус "На совещании"
        if status == UserStatus.MEETING:
            return self.responses["meeting"]
        # Статус "На обеде"
        if status == UserStatus.LUNCH:
            return self.responses["lunch"].format(time="14:00")
        # Режим "По рабочим часам"
        if mode == AutoReplyMode.WORK_HOURS:
            tz = pytz.timezone(params.get("timezone", "Europe/Moscow"))
            now = datetime.now(tz)
            # Выходные
            if now.weekday() >= 5:  # суббота или воскресенье
                return self.responses["weekend"]
            # Ночное время (22:00 - 9:00)
            if now.hour >= 22 or now.hour < 9:
                return self.responses["night"]
            # Обеденное время (13:00 - 14:00)
            if 13 <= now.hour < 14:
                return self.responses["lunch"].format(time="14:00")
            # Рабочие часы (9:00 - 18:00) - НЕ отвечаем автоответчиком
            if 9 <= now.hour < 18:
                return None
            # Вечернее время (18:00 - 22:00)
            return self.responses["work_hours"]
        # Режим "Всегда"
        if mode == AutoReplyMode.ALWAYS:
            return self.responses["default"]
        return None

    def should_auto_reply(self, mode: AutoReplyMode, status: UserStatus,
                        params: Dict = None) -> bool:
        """Определяет, нужно ли отправлять автоответ"""
        if mode == AutoReplyMode.OFF:
            return False
        # Всегда автоответ для специальных статусов
        if status in [UserStatus.VACATION, UserStatus.SICK, UserStatus.MEETING]:
            return True
        # Режим "Всегда включен"
        if mode == AutoReplyMode.ALWAYS:
            return True
        # Режим "По рабочим часам"
        if mode == AutoReplyMode.WORK_HOURS:
            tz = pytz.timezone(params.get("timezone", "Europe/Moscow"))
            now = datetime.now(tz)
            # Рабочие часы (9:00-18:00) - НЕ автоответ
            if 9 <= now.hour < 18 and now.weekday() < 5:
                return False
            return True
        # Режим "Пользовательский"
        if mode == AutoReplyMode.CUSTOM:
            return True
        return False

# ==================== ОСНОВНОЙ КЛАСС БОТА ====================
class MaryAssistantBot:
    def __init__(self, telegram_token: str, deepseek_key: str):
        self.token = telegram_token
        self.ai = DeepSeekAI(deepseek_key)
        self.auto_reply = AutoReplyManager()
        # База данных
        self.db_file = "mary_database.json"
        self.load_database()
        # Системный промпт для Мани
        self.system_prompt = """Ты Маня — профессиональный, но дружелюбный секретарь. Твой стиль:
👩‍💼 *Профессионализм:* точность, пунктуальность, внимание к деталям
💖 *Дружелюбие:* теплое, поддерживающее отношение
🗂️ *Организованность:* всё по полочкам, ничего не забывается
🎯 *Эффективность:* решаю задачи быстро и качественно
Твои обязанности:
1. 📅 Управление расписанием и встречами
2. 📝 Ведение списка дел и задач
3. ⏰ Установка напоминаний
4. 💬 Общение с клиентами (дружелюбно, но профессионально)
5. 🤖 Автоответчик в нерабочее время
6. 📋 Организация информации
Твой тон:
• Используй вежливые обращения: "Добрый день", "Будьте добры"
• Будь точной в деталях
• Добавляй эмодзи для теплоты 👩‍💼💕
• Подписывайся: "С уважением, Маня"
Формат ответов: кратко, по делу, с конкретными действиями."""

    # ==================== БАЗА ДАННЫХ ====================
    def load_database(self):
        """Загружает базу данных из файла"""
        try:
            with open(self.db_file, 'r', encoding='utf-8') as f:
                self.db = json.load(f)
        except:
            self.db = {
                "users": {},
                "tasks": {},
                "appointments": {},
                "reminders": {},
                "settings": {}
            }
            self.save_database()

    def save_database(self):
        """Сохраняет базу данных в файл"""
        with open(self.db_file, 'w', encoding='utf-8') as f:
            json.dump(self.db, f, ensure_ascii=False, indent=2)

    def get_user_data(self, user_id: int) -> Dict:
        """Получает или создаёт данные пользователя"""
        user_id_str = str(user_id)
        if user_id_str not in self.db["users"]:
            self.db["users"][user_id_str] = {
                "name": "",
                "timezone": "Europe/Moscow",
                "working_hours": {"start": "09:00", "end": "18:00"},
                "lunch_hours": {"start": "13:00", "end": "14:00"},
                "autoreply_mode": AutoReplyMode.WORK_HOURS.value,
                "status": UserStatus.AVAILABLE.value,
                "custom_autoreply": "👩‍💼 Я сейчас занята. Отвечу вам в ближайшее время!",
                "vacation_start": None,
                "vacation_end": None,
                "sick_until": None,
                "message_count": 0
            }
            self.save_database()
        return self.db["users"][user_id_str]

    def update_user_data(self, user_id: int, data: Dict):  # ИСПРАВЛЕНО: data: Dict
        """Обновляет данные пользователя"""
        user_data = self.get_user_data(user_id)
        user_data.update(data)
        self.save_database()

    # ==================== JOB QUEUE: НАПОМИНАНИЯ ====================
    @staticmethod
    async def send_reminder_job(context: ContextTypes.DEFAULT_TYPE):
        """Функция, вызываемая job queue для отправки напоминания"""
        job = context.job
        user_id = job.data["user_id"]
        text = job.data["text"]
        reminder_id = job.data["reminder_id"]

        try:
            await context.bot.send_message(
                chat_id=user_id,
                text=f"🔔 *Напоминание от Мани:*\n{text}\n👩‍💼 С уважением, Маня",
                parse_mode="Markdown"
            )
            # Обновляем статус напоминания в БД
            bot_instance = context.application.bot_data.get("bot_instance")
            if bot_instance:
                reminders_list = bot_instance.db["reminders"].get(str(user_id), [])
                for reminder in reminders_list:
                    if reminder["id"] == reminder_id:
                        reminder["status"] = "sent"
                bot_instance.save_database()
        except Exception as e:
            logger.error(f"Ошибка отправки напоминания: {e}")

    async def schedule_reminder(self, context: ContextTypes.DEFAULT_TYPE, user_id: int, reminder_id: int, reminder_time: datetime, text: str):
        """Планирует напоминание через JobQueue"""
        context.job_queue.run_once(
            callback=self.send_reminder_job,
            when=reminder_time,
            data={"user_id": user_id, "text": text, "reminder_id": reminder_id},
            name=f"reminder_{user_id}_{reminder_id}"
        )

    def parse_reminder_time(self, text: str) -> datetime:
        """Парсит время для напоминания"""
        now = datetime.now()
        # Ищем время HH:MM
        time_match = re.search(r'(\d{1,2}):(\d{2})', text)
        if time_match:
            hour, minute = int(time_match.group(1)), int(time_match.group(2))
            reminder_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            # Если время уже прошло - на завтра
            if reminder_time <= now:
                reminder_time += timedelta(days=1)
            return reminder_time
        # Завтра в HH:MM
        if "завтра" in text.lower():
            time_match = re.search(r'завтра в (\d{1,2}):(\d{2})', text, re.IGNORECASE)
            if time_match:
                hour, minute = int(time_match.group(1)), int(time_match.group(2))
                return (now + timedelta(days=1)).replace(hour=hour, minute=minute, second=0, microsecond=0)
        # Через N часов/минут
        time_match = re.search(r'через (\d+) (час[аов]?|минут[уы]?)', text, re.IGNORECASE)
        if time_match:
            num = int(time_match.group(1))
            unit = time_match.group(2)
            if 'час' in unit:
                return now + timedelta(hours=num)
            elif 'минут' in unit:
                return now + timedelta(minutes=num)
        # По умолчанию - через 1 час
        return now + timedelta(hours=1)

    async def add_reminder(self, user_id: int, text: str, context: ContextTypes.DEFAULT_TYPE):
        """Добавляет напоминание"""
        user_id_str = str(user_id)
        if user_id_str not in self.db["reminders"]:
            self.db["reminders"][user_id_str] = []
        reminder_time = self.parse_reminder_time(text)
        reminder = {
            "id": len(self.db["reminders"][user_id_str]) + 1,
            "text": text,
            "time": reminder_time.isoformat(),
            "created": datetime.now().isoformat(),
            "status": "active"
        }
        self.db["reminders"][user_id_str].append(reminder)
        self.save_database()
        await self.schedule_reminder(context, user_id, reminder["id"], reminder_time, text)

    # ==================== АВТООТВЕТЧИК ====================
    async def check_auto_reply(self, user_id: int, message: str = "") -> Optional[str]:
        """Проверяет, нужно ли отправить автоответ"""
        user_data = self.get_user_data(user_id)
        try:
            mode = AutoReplyMode(user_data["autoreply_mode"])
            status = UserStatus(user_data["status"])
        except:
            mode = AutoReplyMode.WORK_HOURS
            status = UserStatus.AVAILABLE
        # Параметры для автоответчика
        params = {
            "timezone": user_data["timezone"],
            "custom_message": user_data.get("custom_autoreply"),
            "vacation_end": user_data.get("vacation_end"),
            "sick_until": user_data.get("sick_until")
        }
        # Проверяем, нужно ли отвечать
        if not self.auto_reply.should_auto_reply(mode, status, params):
            return None
        # Получаем ответ
        return self.auto_reply.get_response(mode, status, params, message)

    # ==================== ОСНОВНЫЕ КОМАНДЫ ====================
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /start"""
        user = update.effective_user
        user_id = user.id
        # Сохраняем пользователя
        user_data = self.get_user_data(user_id)
        if not user_data["name"]:
            user_data["name"] = user.first_name
            self.save_database()
        welcome_text = f"""👩‍💼 *Добрый день, {user.first_name}!*
Я — *Маня*, ваш персональный секретарь с искусственным интеллектом и умным автоответчиком.
*Мои возможности:*
🤖 **Умные диалоги** — DeepSeek AI, работает в России
📅 **Планирование встреч** — "Встреча с клиентом завтра в 14:00"
📝 **Управление задачами** — "Добавь задачу: подготовить отчёт"
⏰ **Автонапоминания** — "Напомни позвонить маме в 18:00"
🔔 **Умный автоответчик** — 5 режимов работы
🗂️ **Организация** — всё хранится в порядке
*Автоответчик умеет:*
• 🕐 Отвечать по рабочим часам (после 18:00, выходные)
• 🔔 Всегда отвечать автоматически
• 🏖️ Сообщать об отпуске/больничном
• ✏️ Ваш собственный текст автоответа
• ❌ Быть выключенным (отвечаю лично)
*Быстрые команды:*
/autoreply — настройки автоответчика
/status — изменить свой статус
/tasks — список задач
/today — что на сегодня
/help — помощь
*Или просто напишите мне что угодно — я всё пойму!* 👩‍💼"""
        keyboard = [
            [
                InlineKeyboardButton("🤖 Автоответчик", callback_data="autoreply_menu"),
                InlineKeyboardButton("📅 Сегодня", callback_data="today")
            ],
            [
                InlineKeyboardButton("📝 Задачи", callback_data="tasks"),
                InlineKeyboardButton("⏰ Напоминания", callback_data="reminders")
            ],
            [
                InlineKeyboardButton("🏖️ Отпуск", callback_data="vacation"),
                InlineKeyboardButton("🤒 Больничный", callback_data="sick")
            ],
            [
                InlineKeyboardButton("⚙️ Настройки", callback_data="settings"),
                InlineKeyboardButton("❓ Помощь", callback_data="help")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            welcome_text,
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def autoreply_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /autoreply"""
        await self.show_autoreply_menu(update)

    async def show_autoreply_menu(self, update: Update):
        """Показывает меню автоответчика"""
        user_id = update.effective_user.id
        user_data = self.get_user_data(user_id)
        mode = AutoReplyMode(user_data["autoreply_mode"])
        status = UserStatus(user_data["status"])
        mode_names = {
            AutoReplyMode.OFF: "❌ Выключен",
            AutoReplyMode.WORK_HOURS: "🕐 По рабочим часам",
            AutoReplyMode.ALWAYS: "🔔 Всегда включен",
            AutoReplyMode.CUSTOM: "✏️ Пользовательский",
            AutoReplyMode.VACATION: "🏖️ Отпуск",
            AutoReplyMode.SICK: "🤒 Больничный"
        }
        status_names = {
            UserStatus.AVAILABLE: "🟢 Доступна",
            UserStatus.BUSY: "🟡 Занята",
            UserStatus.MEETING: "🟠 На совещании",
            UserStatus.VACATION: "🏖️ В отпуске",
            UserStatus.SICK: "🤒 Болею",
            UserStatus.LUNCH: "🍽️ На обеде"
        }
        text = f"""👩‍💼 *Настройки автоответчика*
*Текущий режим:* {mode_names[mode]}
*Ваш статус:* {status_names[status]}
*Режимы работы:*
🕐 *По рабочим часам* — автоответ после 18:00, в выходные и ночью
🔔 *Всегда включен* — автоответ на все сообщения
✏️ *Пользовательский* — ваш собственный текст
🏖️ *Отпуск* — с указанием даты возвращения
🤒 *Больничный* — с датой выздоровления
❌ *Выключен* — всегда отвечаю лично
*Нажмите кнопку для изменения:*"""
        keyboard = [
            [
                InlineKeyboardButton("🕐 Рабочие часы", callback_data="ar_work"),
                InlineKeyboardButton("🔔 Всегда", callback_data="ar_always")
            ],
            [
                InlineKeyboardButton("✏️ Свой текст", callback_data="ar_custom"),
                InlineKeyboardButton("❌ Выключить", callback_data="ar_off")
            ],
            [
                InlineKeyboardButton("🏖️ В отпуск", callback_data="status_vacation"),
                InlineKeyboardButton("🤒 Больничный", callback_data="status_sick")
            ],
            [
                InlineKeyboardButton("🟡 Занята", callback_data="status_busy"),
                InlineKeyboardButton("🟢 Доступна", callback_data="status_available")
            ],
            [
                InlineKeyboardButton("🔙 Назад", callback_data="back_to_main")
            ]
        ]
        if hasattr(update, 'callback_query'):
            await update.callback_query.edit_message_text(
                text=text,
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text(
                text=text,
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="Markdown"
            )

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /status"""
        if not context.args:
            text = """👩‍💼 *Изменение статуса*
Используйте:
/status available — 🟢 Доступна
/status busy — 🟡 Занята
/status meeting — 🟠 На совещании
/status vacation ДД.ММ ДД.ММ — 🏖️ Отпуск (пример: /status vacation 15.01 25.01)
/status sick ДД.ММ — 🤒 Больничный (пример: /status sick 20.01)
Статус влияет на автоответчик!"""
            await update.message.reply_text(text, parse_mode="Markdown")
            return
        user_id = update.effective_user.id
        status_arg = context.args[0].lower()
        if status_arg == "available":
            self.update_user_data(user_id, {
                "status": UserStatus.AVAILABLE.value,
                "autoreply_mode": AutoReplyMode.WORK_HOURS.value
            })
            await update.message.reply_text("👩‍💼 Статус изменён на 'Доступна'. Автоответчик по рабочим часам.")
        elif status_arg == "busy":
            self.update_user_data(user_id, {
                "status": UserStatus.BUSY.value,
                "autoreply_mode": AutoReplyMode.ALWAYS.value
            })
            await update.message.reply_text("👩‍💼 Статус 'Занята'. Включён автоответчик.")
        elif status_arg == "meeting":
            self.update_user_data(user_id, {
                "status": UserStatus.MEETING.value,
                "autoreply_mode": AutoReplyMode.ALWAYS.value
            })
            await update.message.reply_text("👩‍💼 Статус 'На совещании'. Включён автоответчик.")
        elif status_arg == "vacation":
            if len(context.args) >= 3:
                start_date, end_date = context.args[1], context.args[2]
                self.update_user_data(user_id, {
                    "status": UserStatus.VACATION.value,
                    "autoreply_mode": AutoReplyMode.VACATION.value,
                    "vacation_start": start_date,
                    "vacation_end": end_date
                })
                await update.message.reply_text(f"👩‍💼 Статус 'В отпуске' с {start_date} по {end_date}. Автоответчик включён.")
            else:
                await update.message.reply_text("👩‍💼 Укажите даты отпуска: /status vacation ДД.ММ ДД.ММ")
        elif status_arg == "sick":
            if len(context.args) >= 2:
                return_date = context.args[1]
                self.update_user_data(user_id, {
                    "status": UserStatus.SICK.value,
                    "autoreply_mode": AutoReplyMode.SICK.value,
                    "sick_until": return_date
                })
                await update.message.reply_text(f"👩‍💼 Статус 'Болею' до {return_date}. Автоответчик включён.")
            else:
                await update.message.reply_text("👩‍💼 Укажите дату возвращения: /status sick ДД.ММ")

    async def tasks_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /tasks"""
        user_id = update.effective_user.id
        user_tasks = self.db["tasks"].get(str(user_id), [])
        if not user_tasks:
            text = "👩‍💼 *У вас пока нет задач!*\nДобавьте задачу, написав мне:\n• \"Нужно сделать отчёт к пятнице\"\n• \"Задача: купить продукты\"\n• Или просто скажите что нужно сделать"
        else:
            text = "👩‍💼 *Ваши задачи:*\n"
            for i, task in enumerate(user_tasks[:10], 1):
                status = "✅" if task.get("completed", False) else "⏳"
                text += f"{i}. {status} {task['text'][:50]}"
                if len(task['text']) > 50:
                    text += "..."
                text += "\n"
        await update.message.reply_text(text, parse_mode="Markdown")

    async def today_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /today"""
        user_id = update.effective_user.id
        user_data = self.get_user_data(user_id)
        today = datetime.now().strftime("%d.%m.%Y")
        text = f"👩‍💼 *Ваш день на {today}:*\n"
        # Получаем встречи
        appointments = self.db["appointments"].get(str(user_id), [])
        today_appointments = []
        for app in appointments:
            created = datetime.fromisoformat(app.get("created", datetime.now().isoformat()))
            if created.date() == datetime.now().date():
                today_appointments.append(app)
        if today_appointments:
            text += "*📅 Сегодняшние встречи:*\n"
            for app in today_appointments[:5]:
                time = app.get("time", "время не указано")
                text += f"• ⏰ {time} - {app['text'][:40]}...\n"
        # Получаем задачи
        tasks = self.db["tasks"].get(str(user_id), [])
        active_tasks = [t for t in tasks if not t.get("completed", False)]
        if active_tasks:
            text += "\n*📝 Активные задачи:*\n"
            for task in active_tasks[:5]:
                text += f"• ⏳ {task['text'][:40]}...\n"
        if not today_appointments and not active_tasks:
            text += "🎉 *Свободный день!*\nОтличное время для отдыха или планирования!"
        # Добавляем информацию об автоответчике
        mode = AutoReplyMode(user_data["autoreply_mode"])
        status = UserStatus(user_data["status"])
        if mode != AutoReplyMode.OFF:
            auto_reply_text = await self.check_auto_reply(user_id)
            if auto_reply_text:
                text += f"\n*🤖 Автоответчик:* Включён ({mode.value})"
        text += "\nС уважением, Маня 👩‍💼"
        await update.message.reply_text(text, parse_mode="Markdown")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /help"""
        help_text = """👩‍💼 *Помощь по боту Маня*
*Основные команды:*
/start — главное меню
/autoreply — настройки автоответчика
/status — изменить статус (available, busy, meeting, vacation, sick)
/tasks — список задач
/today — что на сегодня
/help — эта справка
*Автоответчик:*
• Работает по расписанию (после 18:00, выходные, ночью)
• Может быть всегда включён
• Сообщает об отпуске/больничном
• Можно настроить свой текст
*Как общаться:*
1. Пишите *естественным языком*
2. Указывайте *время и даты*
3. Я *сама пойму* что вам нужно
*Примеры:*
• "Запланируй встречу с клиентом завтра в 14:00"
• "Напомни позвонить маме в 18:00"
• "Добавь задачу: подготовить отчёт к пятнице"
• "Что у меня на завтра?"
*С уважением, ваша Маня* 👩‍💼"""
        await update.message.reply_text(help_text, parse_mode="Markdown")

    # ==================== ОБРАБОТКА СООБЩЕНИЙ ====================
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка всех текстовых сообщений"""
        user = update.effective_user
        user_id = user.id
        user_text = update.message.text
        # Обновляем счётчик сообщений
        user_data = self.get_user_data(user_id)
        user_data["message_count"] = user_data.get("message_count", 0) + 1
        self.save_database()
        # Проверяем автоответчик
        auto_reply = await self.check_auto_reply(user_id, user_text)
        if auto_reply:
            await update.message.reply_text(auto_reply, parse_mode="Markdown")
            return
        # Показываем что Маня думает
        thinking_msg = await update.message.reply_text("👩‍💼 Думаю...")
        try:
            # Формируем контекст для ИИ
            user_context = f"""
Имя пользователя: {user.first_name}
Текущее время: {datetime.now().strftime('%H:%M')}
Сообщение пользователя: {user_text}
"""
            # Запрос к DeepSeek AI
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "system", "content": user_context},
                {"role": "user", "content": user_text}
            ]
            ai_response = await self.ai.chat(messages)
            # Если ИИ не ответил - используем резервный ответ
            if not ai_response:
                ai_response = "👩‍💼 Простите, у меня временные технические сложности. Можете повторить запрос?"
            # Удаляем сообщение "думаю"
            await thinking_msg.delete()
            # Отправляем ответ
            await update.message.reply_text(
                f"{ai_response}\n👩‍💼 С уважением, Маня",
                parse_mode="Markdown"
            )
            # Автоматически обрабатываем задачи и встречи
            await self.auto_process_request(user_id, user_text, ai_response, context)
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await thinking_msg.delete()
            await update.message.reply_text(
                "👩‍💼 Простите, произошла ошибка. Попробуйте ещё раз или используйте команды из меню.",
                parse_mode="Markdown"
            )

    async def auto_process_request(self, user_id: int, user_text: str, ai_response: str, context: ContextTypes.DEFAULT_TYPE):
        """Автоматически обрабатывает запросы на задачи и встречи"""
        text_lower = user_text.lower()
        # Определяем задачу
        task_keywords = ["задача", "сделать", "нужно", "надо", "поручение", "дело"]
        if any(word in text_lower for word in task_keywords):
            await self.add_task(user_id, user_text)
        # Определяем встречу
        meeting_keywords = ["встреча", "совещание", "встречу", "конференция", "звонок"]
        if any(word in text_lower for word in meeting_keywords):
            await self.add_appointment(user_id, user_text)
        # Определяем напоминание
        reminder_keywords = ["напомни", "напоминание", "напомнить"]
        if any(word in text_lower for word in reminder_keywords):
            await self.add_reminder(user_id, user_text, context)

    async def add_task(self, user_id: int, text: str):
        """Добавляет задачу"""
        user_id_str = str(user_id)
        if user_id_str not in self.db["tasks"]:
            self.db["tasks"][user_id_str] = []
        task = {
            "id": len(self.db["tasks"][user_id_str]) + 1,
            "text": text,
            "created": datetime.now().isoformat(),
            "completed": False
        }
        self.db["tasks"][user_id_str].append(task)
        self.save_database()

    async def add_appointment(self, user_id: int, text: str):
        """Добавляет встречу"""
        user_id_str = str(user_id)
        if user_id_str not in self.db["appointments"]:
            self.db["appointments"][user_id_str] = []
        # Парсим время
        time_match = re.search(r'(\d{1,2}):(\d{2})', text)
        appointment = {
            "id": len(self.db["appointments"][user_id_str]) + 1,
            "text": text,
            "created": datetime.now().isoformat(),
            "time": time_match.group(0) if time_match else "не указано"
        }
        self.db["appointments"][user_id_str].append(appointment)
        self.save_database()

    # ==================== ОБРАБОТКА КНОПОК ====================
    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка нажатий кнопок"""
        query = update.callback_query
        await query.answer()
        data = query.data
        if data == "autoreply_menu":
            await self.show_autoreply_menu(update)
        elif data == "today":
            await self.today_from_button(update)
        elif data == "tasks":
            await self.tasks_from_button(update)
        elif data == "reminders":
            await self.reminders_from_button(update)
        elif data == "vacation":
            await self.vacation_dialog(update)
        elif data == "sick":
            await self.sick_dialog(update)
        elif data == "settings":
            await self.settings_dialog(update)
        elif data == "help":
            await self.help_dialog(update)
        elif data == "back_to_main":
            await self.back_to_main(update)
        # Обработка автоответчика
        elif data.startswith("ar_"):
            mode = data[3:]  # work, always, custom, off
            await self.set_autoreply_mode(update, mode)
        # Обработка статусов
        elif data.startswith("status_"):
            status = data[7:]  # vacation, sick, busy, available
            await self.set_status_from_button(update, status)

    async def set_autoreply_mode(self, update: Update, mode: str):
        """Устанавливает режим автоответчика"""
        query = update.callback_query
        user_id = query.from_user.id  # Используем query.from_user
        user_data = self.get_user_data(user_id)
        mode_map = {
            "work": AutoReplyMode.WORK_HOURS,
            "always": AutoReplyMode.ALWAYS,
            "custom": AutoReplyMode.CUSTOM,
            "off": AutoReplyMode.OFF
        }
        if mode in mode_map:
            user_data["autoreply_mode"] = mode_map[mode].value
            self.save_database()
            messages = {
                "work": "👩‍💼 Автоответчик включён по рабочим часам (после 18:00, выходные, ночью)",
                "always": "👩‍💼 Автоответчик всегда включён",
                "custom": "👩‍💼 Включён пользовательский автоответчик",
                "off": "👩‍💼 Автоответчик выключен. Отвечаю на все сообщения лично"
            }
            await query.edit_message_text(messages[mode])
            await asyncio.sleep(2)
            await self.show_autoreply_menu(update)

    async def set_status_from_button(self, update: Update, status: str):
        """Устанавливает статус из кнопки"""
        query = update.callback_query
        user_id = query.from_user.id
        if status == "vacation":
            await query.edit_message_text(
                "👩‍💼 *Настройка отпуска*\nНапишите даты в формате:\n/status vacation ДД.ММ ДД.ММ\n*Пример:* /status vacation 15.01 25.01"
            )
        elif status == "sick":
            await query.edit_message_text(
                "👩‍💼 *Настройка больничного*\nНапишите дату возвращения:\n/status sick ДД.ММ\n*Пример:* /status sick 20.01"
            )
        elif status == "busy":
            self.update_user_data(user_id, {
                "status": UserStatus.BUSY.value,
                "autoreply_mode": AutoReplyMode.ALWAYS.value
            })
            await query.edit_message_text("👩‍💼 Статус изменён на 'Занята'. Включён автоответчик.")
        elif status == "available":
            self.update_user_data(user_id, {
                "status": UserStatus.AVAILABLE.value,
                "autoreply_mode": AutoReplyMode.WORK_HOURS.value
            })
            await query.edit_message_text("👩‍💼 Статус изменён на 'Доступна'. Автоответчик по рабочим часам.")

    async def today_from_button(self, update: Update):
        """Показывает сегодняшний день из кнопки"""
        query = update.callback_query
        user_id = query.from_user.id
        today = datetime.now().strftime("%d.%m.%Y")
        text = f"👩‍💼 *Сегодня {today}:*\n"
        text += "📅 *Встречи:*\n• Нет запланированных встреч\n"
        text += "📝 *Задачи:*\n• Добавьте задачи, написав мне\n"
        text += "🎉 *Совет дня:* Отличное время для планирования следующей недели!\n"
        text += "👩‍💼 С уважением, Маня"
        await query.edit_message_text(text, parse_mode="Markdown")

    async def tasks_from_button(self, update: Update):
        """Показывает задачи из кнопки"""
        query = update.callback_query
        user_id = query.from_user.id
        user_tasks = self.db["tasks"].get(str(user_id), [])
        if not user_tasks:
            text = "👩‍💼 *У вас пока нет задач!*\nДобавьте задачу, написав мне:\n• \"Нужно сделать отчёт к пятнице\"\n• \"Задача: купить продукты\"\n• Или просто скажите что нужно сделать"
        else:
            text = "👩‍💼 *Ваши задачи:*\n"
            for i, task in enumerate(user_tasks[:10], 1):
                status = "✅" if task.get("completed", False) else "⏳"
                text += f"{i}. {status} {task['text'][:50]}"
                if len(task['text']) > 50:
                    text += "..."
                text += "\n"
        await query.edit_message_text(text, parse_mode="Markdown")

    async def reminders_from_button(self, update: Update):
        """Показывает напоминания"""
        query = update.callback_query
        user_id = query.from_user.id
        reminders = self.db["reminders"].get(str(user_id), [])
        if not reminders:
            text = "👩‍💼 *У вас пока нет напоминаний!*\nНапишите, например:\n• \"Напомни позвонить маме в 18:00\"\n• \"Напоминание: оплатить счет завтра\""
        else:
            text = "👩‍💼 *Ваши напоминания:*\n"
            for i, rem in enumerate(reminders[-5:], 1):
                time_str = datetime.fromisoformat(rem["time"]).strftime("%d.%m %H:%M")
                status = "✅" if rem["status"] == "sent" else "⏰"
                text += f"{i}. {status} {time_str} — {rem['text'][:40]}...\n"
        await query.edit_message_text(text, parse_mode="Markdown")

    async def vacation_dialog(self, update: Update):
        """Диалог настройки отпуска"""
        query = update.callback_query
        text = """👩‍💼 *Настройка отпуска*
Чтобы уйти в отпуск, напишите:
/status vacation ДД.ММ ДД.ММ
*Пример:*
/status vacation 15.01 25.01
*Что произойдёт:*
1. Статус изменится на "В отпуске"
2. Включится автоответчик с датой возвращения
3. Все будут знать, что вы в отпуске
*Для выхода из отпуска:*
/status available"""
        await query.edit_message_text(text, parse_mode="Markdown")

    async def sick_dialog(self, update: Update):
        """Диалог настройки больничного"""
        query = update.callback_query
        text = """👩‍💼 *Настройка больничного*
Чтобы уйти на больничный, напишите:
/status sick ДД.ММ
*Пример:*
/status sick 20.01
*Что произойдёт:*
1. Статус изменится на "Болею"
2. Включится автоответчик с датой возвращения
3. Все будут знать, что вы на больничном
*Для выхода с больничного:*
/status available"""
        await query.edit_message_text(text, parse_mode="Markdown")

    async def settings_dialog(self, update: Update):
        """Диалог настроек"""
        query = update.callback_query
        text = """👩‍💼 *Настройки бота*
Сейчас доступны только настройки через команды:
• /autoreply — автоответчик
• /status — статус
В будущих версиях появятся дополнительные настройки!
👩‍💼 С уважением, Маня"""
        await query.edit_message_text(text, parse_mode="Markdown")

    async def help_dialog(self, update: Update):
        """Диалог помощи"""
        query = update.callback_query
        text = """👩‍💼 *Помощь и поддержка*
*Частые вопросы:*
1. *Как настроить автоответчик?*
Используйте /autoreply или кнопку "🤖 Автоответчик"
2. *Как добавить задачу?*
Просто напишите: "Нужно сделать отчёт" или "Задача: купить продукты"
3. *Как установить напоминание?*
Напишите: "Напомни позвонить маме в 18:00"
4. *Как запланировать встречу?*
Напишите: "Встреча с клиентом завтра в 14:00"
5. *Бот не отвечает?*
• Проверьте, не включён ли автоответчик
• Попробуйте команду /start
• Напишите мне снова
*Для быстрой помощи пишите:* @mary_secretary_bot
👩‍💼 С уважением, Маня"""
        await query.edit_message_text(text, parse_mode="Markdown")

    async def back_to_main(self, update: Update):
        """Возврат в главное меню"""
        query = update.callback_query
        user = query.from_user
        text = f"""👩‍💼 *Главное меню*
Выберите действие, {user.first_name}:"""
        keyboard = [
            [
                InlineKeyboardButton("🤖 Автоответчик", callback_data="autoreply_menu"),
                InlineKeyboardButton("📅 Сегодня", callback_data="today")
            ],
            [
                InlineKeyboardButton("📝 Задачи", callback_data="tasks"),
                InlineKeyboardButton("⏰ Напоминания", callback_data="reminders")
            ],
            [
                InlineKeyboardButton("🏖️ Отпуск", callback_data="vacation"),
                InlineKeyboardButton("🤒 Больничный", callback_data="sick")
            ],
            [
                InlineKeyboardButton("⚙️ Настройки", callback_data="settings"),
                InlineKeyboardButton("❓ Помощь", callback_data="help")
            ]
        ]
        await query.edit_message_text(
            text=text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )

    # ==================== ЗАПУСК БОТА В РЕЖИМЕ WEBHOOK ====================
    def run(self):
        """Запускает бота в режиме webhook для Render"""
        application = Application.builder().token(self.token).build()
        # Сохраняем ссылку на экземпляр бота для доступа из job
        application.bot_data["bot_instance"] = self

        # Регистрируем обработчики
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("autoreply", self.autoreply_command))
        application.add_handler(CommandHandler("status", self.status_command))
        application.add_handler(CommandHandler("tasks", self.tasks_command))
        application.add_handler(CommandHandler("today", self.today_command))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CallbackQueryHandler(self.button_handler))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

        print("=" * 70)
        print("👩‍💼 ЗАПУСКАЕТСЯ СЕКРЕТАРЬ МАНЯ В РЕЖИМЕ WEBHOOK")
        print("=" * 70)
        print("🤖 ИИ: DeepSeek API (работает в России)")
        print("🔔 Автоответчик: 5 режимов работы")
        print("📅 Умное планирование встреч и задач")
        print("⏰ Интеллектуальные напоминания")
        print("☁️  Готов к работе в облаке 24/7")
        print("=" * 70)
        print("\n📱 Открой Telegram и напиши боту /start")
        print("🌍 Бот будет работать ВЕЗДЕ без твоего компьютера")
        print("=" * 70)

        # Получаем URL сервиса из переменной окружения RENDER_EXTERNAL_URL
        webhook_url = os.getenv("RENDER_EXTERNAL_URL")
        if not webhook_url:
            raise ValueError("❌ ОШИБКА: RENDER_EXTERNAL_URL не задан! Укажите его в Environment Variables на Render.")

        # Безопасный путь webhook (используем часть токена)
        webhook_path = f"/webhook/{self.token.split(':')[1]}"
        full_webhook_url = webhook_url.rstrip('/') + webhook_path

        # Запускаем webhook
        application.run_webhook(
            listen="0.0.0.0",
            port=int(os.environ.get("PORT", 10000)),
            url_path=webhook_path,
            webhook_url=full_webhook_url
        )

# ==================== ГЛАВНАЯ ФУНКЦИЯ ====================
def main():
    """Главная функция запуска"""
    load_dotenv()
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    if not TELEGRAM_TOKEN or not DEEPSEEK_API_KEY:
        print("❌ ОШИБКА: Не найдены необходимые ключи!")
        print("\n📋 Проверьте Environment Variables на Render:")
        print("=" * 50)
        print("TELEGRAM_TOKEN=ваш_токен_от_BotFather")
        print("DEEPSEEK_API_KEY=ваш_ключ_deepseek")
        print("=" * 50)
        return

    print(f"✅ Токен Telegram: {TELEGRAM_TOKEN[:15]}...")
    print(f"✅ Ключ DeepSeek: {DEEPSEEK_API_KEY[:15]}...")

    bot = MaryAssistantBot(TELEGRAM_TOKEN, DEEPSEEK_API_KEY)
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\n🛑 Бот остановлен пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")

if __name__ == "__main__":
    main()
