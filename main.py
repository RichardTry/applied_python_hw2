# Второе домашнее задание — написать телеграм-бота
# с помощью aiogram / иной асинхронной библиотеки.
# В качестве задания можно выбрать написание интерфейса
# для взаимодействия с пользователем в рамках вашего годового проекта.
# Можно написать бот для какой-нибудь другой задачи,
# которая вам интересна и связана с ML.
# Самое простое — в качестве функциональности реализовать инференс вашей модели
# (например, для одного объекта и для батча данных),
# выводить подсказки по использованию сервиса/модели,
# попросить оставить отзыв вашему сервису (с помощью кнопок, например)
# и использовать средний рейтинг в статистике,
# реализовать запрос статистики использования сервиса,
# распределения чего-либо и т.п.
# Всего нужно реализовать не менее пяти различных
# по логике и функционалу методов на оценку 10.

import asyncio
import logging
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters.command import Command
from config import TOKEN

# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)
# Объект бота
bot = Bot(token=TOKEN)
# Диспетчер
dp = Dispatcher()

# состояния бота
JUST_STARTED = 0
GENERAL_FEATURE_SELECTION = 1
WAITING_AGE = 2
WAITING_GENDER = 3

users_states = dict()

features_map = {
    'age': 'Возраст',
    'gender': 'Пол',
    'education': 'Образование',
    'maritalStatus': 'Семейное положение'
}

feature_values = {
    'age': None,
    'gender': ['Мужской', 'Женский'],
    'education': ['Среднее специальное',
                  'Среднее', 'Высшее',
                  'Неоконченное высшее',
                  'Неполное среднее',
                  'Два и более высших образования',
                  'Ученая степень'],
    'maritalStatus': ['Не состоял в браке',
                       'Гражданский брак',
                       'Состою в браке',
                       'Разведен(а)',
                       'Вдовец/Вдова']
}

features = list(features_map)

### Управление состояниями

def init_user(user_id):
    users_states[user_id] = {'state': JUST_STARTED,
                             'features': {f: None for f in features}}

def set_user_state(user_id, state):
    print(f'Setting {state} for {user_id}')
    if user_id in users_states:
        users_states[user_id]['state'] = state
    else:
        users_states[user_id] = {'state': state}

def get_user_state(user_id):
    return users_states[user_id].get('state', None)

### Кнопки

def gen_keyboard(*buttons):
    kb = [[]]
    for button in buttons:
        kb[0].append(types.KeyboardButton(text=button))
    return kb

def gen_keyboard_markup(keyboard, hint='Выбери значение'):
    return types.ReplyKeyboardMarkup(
        keyboard=keyboard,
        resize_keyboard=True,
        input_field_placeholder=hint
    )

def gen_feature_string(user_id, feature_key):
    return ('✅ ' if users_states[user_id]['features'][feature_key] is not None else '') + features_map.get(feature_key, 'unknown')

def gen_features_keyboard(user_id):
    buttons = []
    ROW_SIZE = 2
    for r in range(0, len(features), ROW_SIZE):
        row = []
        for c in range(ROW_SIZE):
            index = r+c
            row.append(types.InlineKeyboardButton(text=gen_feature_string(user_id, features[index]), callback_data=f'select_{features[index]}'))
        buttons.append(row)
    all_filled = True
    for f in users_states[user_id]['features']:
        if users_states[user_id]['features'][f] is None:
            all_filled = False
    if all_filled:
        buttons.append([types.InlineKeyboardButton(text='✨ Предсказать!', callback_data=f'predict')])
    keyboard = types.InlineKeyboardMarkup(inline_keyboard=buttons)
    return keyboard

def gen_select_keyboard(feature):
    buttons = []
    for index, val in enumerate(feature_values[feature]):
        print(f'{val} ===== {feature}')
        buttons.append([types.InlineKeyboardButton(text=val, callback_data=f"selected_{feature}_{index}")])
    buttons.append([types.InlineKeyboardButton(text="↩️ Назад", callback_data="back")])
    keyboard = types.InlineKeyboardMarkup(inline_keyboard=buttons)
    return keyboard

async def show_features_keyboard(user_id, message):
    await message.edit_reply_markup(reply_markup=gen_features_keyboard(user_id))

### Коллбеки

@dp.callback_query(F.data.startswith("select_"))
async def callbacks_select(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    feature = callback.data[7:]
    print(f'User selected {feature}')
    if feature_values[feature] is not None: # Если категориальная фича
        print('Cat feature')
        await callback.message.edit_reply_markup(reply_markup=gen_select_keyboard(feature))
    else:
        print('Numeric feature')
        await callback.answer(f'Введите значение признака {feature.lower()}')
        set_user_state(user_id, f'waiting_for_{feature}')

@dp.callback_query(F.data.startswith("selected_"))
async def callbacks_selected(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    feature = callback.data.split("_")[1]
    value = feature_values[feature][int(callback.data.split("_")[2])]
    print(f'User selected {value} in {feature}')
    users_states[user_id]['features'][feature] = value
    await callback.answer(f'Значение {value} выбрано!')
    await show_features_keyboard(user_id, callback.message)

@dp.callback_query(F.data == "back")
async def callbacks_back(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    set_user_state(user_id, GENERAL_FEATURE_SELECTION)
    await show_features_keyboard(user_id, callback.message)

@dp.callback_query(F.data == "predict")
async def callbacks_back(callback: types.CallbackQuery):
    user_input_df = users_states[user_id]['features']

    train_df = open_data()
    train_X_df, _ = split_data(train_df)
    write_user_data(train_X_df)
    full_X_df = pd.concat((user_input_df, train_X_df), axis=0)
    preprocessed_X_df = preprocess_data(full_X_df, test=False)

    user_X_df = preprocessed_X_df[:1]
    write_user_data(user_X_df)

    prediction, prediction_probas = load_model_and_predict(user_X_df)
    write_prediction(prediction, prediction_probas)

### Помощь

def print_help():
    pass

### Команды

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    if message.from_user.id not in users_states:
        init_user(message.from_user.id)
    await message.answer("Привет! Используй команду /predict чтобы получить предсказание.")

@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    await message.answer("Справка по боту: <надо дописать>")

@dp.message(Command("predict"))
async def cmd_predict(message: types.Message):
    user_id = message.from_user.id
    await message.answer("Выберите признак, чтобы заполнить:",
                         reply_markup=gen_features_keyboard(user_id))
    set_user_state(user_id, WAITING_GENDER)

@dp.message()
async def plant_text(message: types.Message):
    user_id = message.from_user.id
    state = get_user_state(message.from_user.id)
    print(f'state = {state}')
    if state == JUST_STARTED:
        print_help()
    elif state == GENERAL_FEATURE_SELECTION:
        message.answer('Выбери признак в таблице выше!')
    elif state == 'waiting_for_age':
        if not message.text.isdigit():
            message.reply('Введён некорректный возраст.\nПопробуйте заново.')
        else:
            print('age entered!')
            users_states[user_id]['features']['age'] = int(message.text)
            await message.answer('Возраст введён!')
            await message.answer("Выберите признак, чтобы заполнить остальные:",
                     reply_markup=gen_features_keyboard(user_id))

stat = [0, 0, 0, 0, 1]
@dp.message(Command("rate"))
async def cmd_rate(message: types.Message):
    await message.answer("Оцени бота от 1 до 5.")

@dp.message(Command("statistic"))
async def cmd_statistic(message: types.Message):
    await message.answer(f"Статистика по оценке бота:\n\
⭐⭐⭐⭐⭐: {stat[4]}\n\
⭐⭐⭐⭐: {stat[3]}\n\
⭐⭐⭐: {stat[2]}\n\
⭐⭐: {stat[1]}\n\
⭐: {stat[0]}\n\
Самая частая оценка: {'⭐' * (stat.index(max(stat)) + 1)}\n\
Средняя оценка: {(stat[0]*1 + stat[1]*2 + stat[2]*3 + stat[3]*4 + stat[4]*5) / sum(stat):.2f}")

# Запуск процесса поллинга новых апдейтов
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())