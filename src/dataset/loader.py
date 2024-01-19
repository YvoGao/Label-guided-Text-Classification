import json
import os

import numpy as np
import pandas as pd
from dataset.utils import InputExample
import datetime

def tprint(s):
    '''
        print datetime and s
        @params:
            s (str): the string to be printed
    '''
    print('{}: {}'.format(
        datetime.datetime.now(), s),
          flush=True)



def get_label_dict(args):
    
    _hwu64_label_dict = {'audio/volume_down': 0, 'audio/volume_mute': 1, 'audio/volume_up': 2, 'calendar/query': 3, 'calendar/remove': 4, 'calendar/set': 5, 'email/addcontact': 6, 'email/query': 7, 'email/querycontact': 8, 'email/sendemail': 9, 'recommendation/events': 10, 'recommendation/locations': 11, 'recommendation/movies': 12, 'takeaway/order': 13, 'takeaway/query': 14, 'transport/query': 15, 'transport/taxi': 16, 'transport/ticket': 17, 'transport/traffic': 18, 'alarm/query': 19, 'alarm/remove': 20, 'alarm/set': 21, 'general/affirm': 22, 'general/commandstop': 23, 'general/confirm': 24, 'general/dontcare': 25, 'general/explain': 26, 'general/joke': 27, 'general/negate': 28, 'general/praise': 29, 'general/quirky': 30, 'general/repeat': 31, 'iot/cleaning': 32, 'iot/coffee': 33, 'iot/hue_lightchange': 34, 'iot/hue_lightdim': 35, 'iot/hue_lightoff': 36, 'iot/hue_lighton': 37, 'iot/hue_lightup': 38, 'iot/wemo_off': 39, 'iot/wemo_on': 40, 'qa/currency': 41, 'qa/definition': 42, 'qa/factoid': 43, 'qa/maths': 44, 'qa/stock': 45, 'social/post': 46, 'social/query': 47, 'weather/query': 48, 'cooking/recipe': 49, 'datetime/convert': 50, 'datetime/query': 51, 'lists/createoradd': 52, 'lists/query': 53, 'lists/remove': 54, 'music/likeness': 55, 'music/query': 56, 'music/settings': 57, 'news/query': 58, 'play/audiobook': 59, 'play/game': 60, 'play/music': 61, 'play/podcasts': 62, 'play/radio': 63}

    _liu_label_dict = {'post': 0, 'locations': 1, 'movies': 2, 'volume_mute': 3, 'radio': 4, 'audiobook': 5, 'stock': 6, 'events': 7, 'recipe': 8, 'game': 9, 'hue_lightdim': 10, 'set': 11, 'traffic': 12, 'definition': 13, 'joke': 14, 'wemo_off': 15, 'commandstop': 16, 'cleaning': 17, 'factoid': 18, 'negate': 19, 'currency': 20, 'hue_lighton': 21, 'coffee': 22, 'confirm': 23, 'wemo_on': 24, 'maths': 25, 'hue_lightup': 26, 'likeness': 27, 'createoradd': 28, 'querycontact': 29, 'repeat': 30, 'hue_lightchange': 31, 'sendemail': 32, 'order': 33, 'ticket': 34, 'convert': 35, 'hue_lightoff': 36, 'podcasts': 37, 'volume_up': 38, 'taxi': 39, 'settings': 40, 'dontcare': 41, 'remove': 42, 'explain': 43, 'dislikeness': 44, 'addcontact': 45, 'volume_down': 46, 'affirm': 47, 'praise': 48, 'greet': 49, 'quirky': 50, 'music': 51, 'query': 52, 'volume_other': 53}

    _reuters_label_dict = {
        'acquisition': 0,
        'aluminium': 1,
        'trade deficit': 2,
        'cocoa': 3,
        'coffee': 4,
        'copper': 5,
        'cotton': 6,
        'inflation': 7,
        'oil': 8,
        'profit': 9,
        'gdp': 10,
        'gold': 11,
        'grain': 12,
        'rate': 13,
        'industrial production': 14,
        'steel': 15,
        'unemployment': 16,
        'cattle': 17,
        'treasury bank': 18,
        'money supply': 19,
        'gas': 20,
        'orange': 21,
        'reserves': 22,
        'retail': 23,
        'rubber': 24,
        'ship': 25,
        'sugar': 26,
        'tin': 27,
        'tariffs': 28,
        'oils and fats tax': 29,
        'producer price wholesale c': 30
    }

    _20news_label_dict = {
        'talk.politics.mideast': 0,
        'sci.space': 1,
        'misc.forsale': 2,
        'talk.politics.misc': 3,
        'comp.graphics': 4,
        'sci.crypt': 5,
        'comp.windows.x': 6,
        'comp.os.ms-windows.misc': 7,
        'talk.politics.guns': 8,
        'talk.religion.misc': 9,
        'rec.autos': 10,
        'sci.med': 11,
        'comp.sys.mac.hardware': 12,
        'sci.electronics': 13,
        'rec.sport.hockey': 14,
        'alt.atheism': 15,
        'rec.motorcycles': 16,
        'comp.sys.ibm.pc.hardware': 17,
        'rec.sport.baseball': 18,
        'soc.religion.christian': 19,
    }

    _amazon_label_dict = {
        'Amazon_Instant_Video': 0,
        'Apps_for_Android': 1,
        'Automotive': 2,
        'Baby': 3,
        'Beauty': 4,
        'Books': 5,
        'CDs_and_Vinyl': 6,
        'Cell_Phones_and_Accessories': 7,
        'Clothing_Shoes_and_Jewelry': 8,
        'Digital_Music': 9,
        'Electronics': 10,
        'Grocery_and_Gourmet_Food': 11,
        'Health_and_Personal_Care': 12,
        'Home_and_Kitchen': 13,
        'Kindle_Store': 14,
        'Movies_and_TV': 15,
        'Musical_Instruments': 16,
        'Office_Products': 17,
        'Patio_Lawn_and_Garden': 18,
        'Pet_Supplies': 19,
        'Sports_and_Outdoors': 20,
        'Tools_and_Home_Improvement': 21,
        'Toys_and_Games': 22,
        'Video_Games': 23
    }

    _huffpost_label_dict = {'POLITICS': 0, 'WELLNESS': 1, 'ENTERTAINMENT': 2, 'TRAVEL': 3, 'STYLE & BEAUTY': 4,
                            'PARENTING': 5, 'HEALTHY LIVING': 6, 'QUEER VOICES': 7, 'FOOD & DRINK': 8, 'BUSINESS': 9,
                            'COMEDY': 10, 'SPORTS': 11, 'BLACK VOICES': 12, 'HOME & LIVING': 13, 'PARENTS': 14,
                            'THE WORLDPOST': 15, 'WEDDINGS': 16, 'WOMEN': 17, 'IMPACT': 18, 'DIVORCE': 19, 'CRIME': 20,
                            'MEDIA': 21, 'WEIRD NEWS': 22, 'GREEN': 23, 'WORLDPOST': 24, 'RELIGION': 25, 'STYLE': 26,
                            'SCIENCE': 27, 'WORLD NEWS': 28, 'TASTE': 29, 'TECH': 30, 'MONEY': 31, 'ARTS': 32,
                            'FIFTY': 33,
                            'GOOD NEWS': 34, 'ARTS & CULTURE': 35, 'ENVIRONMENT': 36, 'COLLEGE': 37,
                            'LATINO VOICES': 38,
                            'CULTURE & ARTS': 39, 'EDUCATION': 40}

    _banking77_label_dict = {'card_arrival': 0, 'card_linking': 1, 'exchange_rate': 2,
                             'card_payment_wrong_exchange_rate': 3, 'extra_charge_on_statement': 4,
                             'pending_cash_withdrawal': 5, 'fiat_currency_support': 6, 'card_delivery_estimate': 7,
                             'automatic_top_up': 8, 'card_not_working': 9, 'exchange_via_app': 10,
                             'lost_or_stolen_card': 11, 'age_limit': 12, 'pin_blocked': 13,
                             'contactless_not_working': 14,
                             'top_up_by_bank_transfer_charge': 15, 'pending_top_up': 16, 'cancel_transfer': 17,
                             'top_up_limits': 18, 'wrong_amount_of_cash_received': 19, 'card_payment_fee_charged': 20,
                             'transfer_not_received_by_recipient': 21, 'supported_cards_and_currencies': 22,
                             'getting_virtual_card': 23, 'card_acceptance': 24, 'top_up_reverted': 25,
                             'balance_not_updated_after_cheque_or_cash_deposit': 26, 'card_payment_not_recognised': 27,
                             'edit_personal_details': 28, 'why_verify_identity': 29, 'unable_to_verify_identity': 30,
                             'get_physical_card': 31, 'visa_or_mastercard': 32, 'topping_up_by_card': 33,
                             'disposable_card_limits': 34, 'compromised_card': 35, 'atm_support': 36,
                             'direct_debit_payment_not_recognised': 37, 'passcode_forgotten': 38,
                             'declined_cash_withdrawal': 39, 'pending_card_payment': 40, 'lost_or_stolen_phone': 41,
                             'request_refund': 42, 'declined_transfer': 43, 'Refund_not_showing_up': 44,
                             'declined_card_payment': 45, 'pending_transfer': 46, 'terminate_account': 47,
                             'card_swallowed': 48, 'transaction_charged_twice': 49, 'verify_source_of_funds': 50,
                             'transfer_timing': 51, 'reverted_card_payment?': 52, 'change_pin': 53,
                             'beneficiary_not_allowed': 54, 'transfer_fee_charged': 55, 'receiving_money': 56,
                             'failed_transfer': 57, 'transfer_into_account': 58, 'verify_top_up': 59,
                             'getting_spare_card': 60, 'top_up_by_cash_or_cheque': 61, 'order_physical_card': 62,
                             'virtual_card_not_working': 63, 'wrong_exchange_rate_for_cash_withdrawal': 64,
                             'get_disposable_virtual_card': 65, 'top_up_failed': 66,
                             'balance_not_updated_after_bank_transfer': 67, 'cash_withdrawal_not_recognised': 68,
                             'exchange_charge': 69, 'top_up_by_card_charge': 70, 'activate_my_card': 71,
                             'cash_withdrawal_charge': 72, 'card_about_to_expire': 73, 'apple_pay_or_google_pay': 74,
                             'verify_my_identity': 75, 'country_support': 76}

    _clinc150_label_dict = {'transfer': 0, 'transactions': 1, 'balance': 2, 'freeze_account': 3, 'pay_bill': 4,
                            'bill_balance': 5, 'bill_due': 6, 'interest_rate': 7, 'routing': 8, 'min_payment': 9,
                            'order_checks': 10, 'pin_change': 11, 'report_fraud': 12, 'account_blocked': 13,
                            'spending_history': 14, 'credit_score': 15, 'report_lost_card': 16, 'credit_limit': 17,
                            'rewards_balance': 18, 'new_card': 19, 'application_status': 20, 'card_declined': 21,
                            'international_fees': 22, 'apr': 23, 'redeem_rewards': 24, 'credit_limit_change': 25,
                            'damaged_card': 26, 'replacement_card_duration': 27, 'improve_credit_score': 28,
                            'expiration_date': 29, 'recipe': 30, 'restaurant_reviews': 31, 'calories': 32,
                            'nutrition_info': 33, 'restaurant_suggestion': 34, 'ingredients_list': 35,
                            'ingredient_substitution': 36, 'cook_time': 37, 'food_last': 38, 'meal_suggestion': 39,
                            'restaurant_reservation': 40, 'confirm_reservation': 41, 'how_busy': 42,
                            'cancel_reservation': 43, 'accept_reservations': 44, 'shopping_list': 45,
                            'shopping_list_update': 46, 'next_song': 47, 'play_music': 48, 'update_playlist': 49,
                            'todo_list': 50, 'todo_list_update': 51, 'calendar': 52, 'calendar_update': 53,
                            'what_song': 54,
                            'order': 55, 'order_status': 56, 'reminder': 57, 'reminder_update': 58, 'smart_home': 59,
                            'traffic': 60, 'directions': 61, 'gas': 62, 'gas_type': 63, 'distance': 64,
                            'current_location': 65, 'mpg': 66, 'oil_change_when': 67, 'oil_change_how': 68,
                            'jump_start': 69, 'uber': 70, 'schedule_maintenance': 71, 'last_maintenance': 72,
                            'tire_pressure': 73, 'tire_change': 74, 'book_flight': 75, 'book_hotel': 76,
                            'car_rental': 77,
                            'travel_suggestion': 78, 'travel_alert': 79, 'travel_notification': 80, 'carry_on': 81,
                            'timezone': 82, 'vaccines': 83, 'translate': 84, 'flight_status': 85,
                            'international_visa': 86,
                            'lost_luggage': 87, 'plug_type': 88, 'exchange_rate': 89, 'time': 90, 'alarm': 91,
                            'share_location': 92, 'find_phone': 93, 'weather': 94, 'text': 95, 'spelling': 96,
                            'make_call': 97, 'timer': 98, 'date': 99, 'calculator': 100, 'measurement_conversion': 101,
                            'flip_coin': 102, 'roll_dice': 103, 'definition': 104, 'direct_deposit': 105,
                            'pto_request': 106, 'taxes': 107, 'payday': 108, 'w2': 109, 'pto_balance': 110,
                            'pto_request_status': 111, 'next_holiday': 112, 'insurance': 113, 'insurance_change': 114,
                            'schedule_meeting': 115, 'pto_used': 116, 'meeting_schedule': 117, 'rollover_401k': 118,
                            'income': 119, 'greeting': 120, 'goodbye': 121, 'tell_joke': 122, 'where_are_you_from': 123,
                            'how_old_are_you': 124, 'what_is_your_name': 125, 'who_made_you': 126, 'thank_you': 127,
                            'what_can_i_ask_you': 128, 'what_are_your_hobbies': 129, 'do_you_have_pets': 130,
                            'are_you_a_bot': 131, 'meaning_of_life': 132, 'who_do_you_work_for': 133, 'fun_fact': 134,
                            'change_ai_name': 135, 'change_user_name': 136, 'cancel': 137, 'user_name': 138,
                            'reset_settings': 139, 'whisper_mode': 140, 'repeat': 141, 'no': 142, 'yes': 143,
                            'maybe': 144,
                            'change_language': 145, 'change_accent': 146, 'change_volume': 147, 'change_speed': 148,
                            'sync_device': 149}
    if args.dataset == '20newsgroup' or args.dataset == '20newsgroup2':
        return _20news_label_dict
    elif args.dataset == 'amazon' or args.dataset == 'amazon2':
        return _amazon_label_dict
    elif args.dataset == 'huffpost':
        return _huffpost_label_dict
    elif args.dataset == 'banking77':
        return _banking77_label_dict
    elif args.dataset == 'clinc150':
        return _clinc150_label_dict
    elif args.dataset == 'reuters':
        return _reuters_label_dict
    elif args.dataset == 'liu':
        return _liu_label_dict
    elif args.dataset == 'hwu64':
        return _hwu64_label_dict


def _get_liu_classes(args):
    '''
        @return list of classes associated with each split
    '''
    label_dict = get_label_dict(args)

    train_classes = list(range(18))
    val_classes = list(range(18, 36))
    test_classes = list(range(36, 54))
    train_class_names, val_class_names, test_class_names = [],[],[]
    for key in label_dict.keys():
        if label_dict[key] in train_classes:
            train_class_names.append(key)
        elif label_dict[key] in test_classes:
            test_class_names.append(key)
        elif label_dict[key] in val_classes:
            val_class_names.append(key)

    return train_classes, val_classes, test_classes, train_class_names, val_class_names, test_class_names



def _get_hwu64_classes(args):
    '''
        @return list of classes associated with each split
    '''
    label_dict = get_label_dict(args)

    train_classes = list(range(23))
    val_classes = list(range(23, 39))
    test_classes = list(range(39, 64))
    train_class_names, val_class_names, test_class_names = [],[],[]
    for key in label_dict.keys():
        if label_dict[key] in train_classes:
            train_class_names.append(key)
        elif label_dict[key] in test_classes:
            test_class_names.append(key)
        elif label_dict[key] in val_classes:
            val_class_names.append(key)

    return train_classes, val_classes, test_classes, train_class_names, val_class_names, test_class_names




def _get_reuters_classes(args):
    '''
        @return list of classes associated with each split
    '''
    label_dict = {
        'acquisition': 0,
        'aluminium': 1,
        'trade deficit': 2,
        'cocoa': 3,
        'coffee': 4,
        'copper': 5,
        'cotton': 6,
        'inflation': 7,
        'oil': 8,
        'profit': 9,
        'gdp': 10,
        'gold': 11,
        'grain': 12,
        'rate': 13,
        'industrial production': 14,
        'steel': 15,
        'unemployment': 16,
        'cattle': 17,
        'treasury bank': 18,
        'money supply': 19,
        'gas': 20,
        'orange': 21,
        'reserves': 22,
        'retail': 23,
        'rubber': 24,
        'ship': 25,
        'sugar': 26,
        'tin': 27,
        'tariffs': 28,
        'oils and fats tax': 29,
        'producer price wholesale c': 30
    }

    train_classes = list(range(15))
    val_classes = list(range(26, 31))
    test_classes = list(range(15, 26))
    train_class_names, val_class_names, test_class_names = [],[],[]
    for key in label_dict.keys():
        if label_dict[key] in train_classes:
            train_class_names.append(key)
        elif label_dict[key] in test_classes:
            test_class_names.append(key)
        elif label_dict[key] in val_classes:
            val_class_names.append(key)

    return train_classes, val_classes, test_classes, train_class_names, val_class_names, test_class_names



def _get_20newsgroup_classes(args):
    '''
        @return list of classes associated with each split
    '''
    label_dict = {
        'talk.politics.mideast': 0,
        'sci.space': 1,
        'misc.forsale': 2,
        'talk.politics.misc': 3,
        'comp.graphics': 4,
        'sci.crypt': 5,
        'comp.windows.x': 6,
        'comp.os.ms-windows.misc': 7,
        'talk.politics.guns': 8,
        'talk.religion.misc': 9,
        'rec.autos': 10,
        'sci.med': 11,
        'comp.sys.mac.hardware': 12,
        'sci.electronics': 13,
        'rec.sport.hockey': 14,
        'alt.atheism': 15,
        'rec.motorcycles': 16,
        'comp.sys.ibm.pc.hardware': 17,
        'rec.sport.baseball': 18,
        'soc.religion.christian': 19,
    }

    
    train_classes = [0, 3, 8, 9, 2, 15, 19, 17]
    val_classes = [4, 6, 7, 12, 18]
    test_classes = [1, 5, 11, 13, 10, 14, 16]



    train_class_names, val_class_names, test_class_names = [], [], []
    for key in label_dict.keys():
        if label_dict[key] in train_classes:
            train_class_names.append(key)
        elif label_dict[key] in test_classes:
            test_class_names.append(key)
        elif label_dict[key] in val_classes:
            val_class_names.append(key)
    return train_classes, val_classes, test_classes, train_class_names, val_class_names, test_class_names


def _get_huffpost_classes(args):
    '''
        @return list of classes associated with each split
    '''
    class_label = ["POLITICS", "WELLNESS", "ENTERTAINMENT", "TRAVEL", "STYLE & BEAUTY", "PARENTING",
                   "HEALTHY LIVING", "QUEER VOICES", "FOOD & DRINK", "BUSINESS", "COMEDY", "SPORTS", "BLACK VOICES",
                   "HOME & LIVING", "PARENTS", "THE WORLDPOST", "WEDDINGS", "WOMEN", "IMPACT", "DIVORCE", "CRIME",
                   "MEDIA", "WEIRD NEWS", "GREEN", "WORLDPOST", "RELIGION", "STYLE", "SCIENCE", "WORLD NEWS", "TASTE",
                   "TECH", "MONEY", "ARTS", "FIFTY", "GOOD NEWS", "ARTS & CULTURE", "ENVIRONMENT", "COLLEGE",
                   "LATINO VOICES",
                   "CULTURE & ARTS", "EDUCATION"]

    train_classes = list(range(20))
    val_classes = list(range(20, 25))
    test_classes = list(range(25, 41))
    train_class_names, val_class_names, test_class_names = [], [], []
    for i in train_classes:
        train_class_names.append(class_label[i])
    for i in val_classes:
        val_class_names.append(class_label[i])
    for i in test_classes:
        train_class_names.append(class_label[i])

    return train_classes, val_classes, test_classes, train_class_names, val_class_names, test_class_names


def _get_amazon_classes(args):
    '''
        @return list of classes associated with each split
    '''
    label_dict = {
        'Amazon_Instant_Video': 0,
        'Apps_for_Android': 1,
        'Automotive': 2,
        'Baby': 3,
        'Beauty': 4,
        'Books': 5,
        'CDs_and_Vinyl': 6,
        'Cell_Phones_and_Accessories': 7,
        'Clothing_Shoes_and_Jewelry': 8,
        'Digital_Music': 9,
        'Electronics': 10,
        'Grocery_and_Gourmet_Food': 11,
        'Health_and_Personal_Care': 12,
        'Home_and_Kitchen': 13,
        'Kindle_Store': 14,
        'Movies_and_TV': 15,
        'Musical_Instruments': 16,
        'Office_Products': 17,
        'Patio_Lawn_and_Garden': 18,
        'Pet_Supplies': 19,
        'Sports_and_Outdoors': 20,
        'Tools_and_Home_Improvement': 21,
        'Toys_and_Games': 22,
        'Video_Games': 23
    }
    train_class_names, val_class_names, test_class_names = [], [], []
    # train_classes = [2, 3, 4, 7, 11, 12, 13, 18, 19, 20]
    # val_classes = [1, 22, 23, 6, 9]
    # test_classes = [0, 5, 14, 15, 8, 10, 16, 17, 21]

    # Meta-SN
    train_classes = [0, 6, 10, 12, 16, 3, 21, 20, 22, 23]
    val_classes = [13, 18, 2, 14, 1]
    test_classes = [7, 17, 9, 11, 19, 4, 15, 5, 8]

    # # TART
    # val_classes = list(range(5))
    # test_classes = list(range(5, 14))
    # train_classes = list(range(14, 24))
    for key in label_dict.keys():
        if label_dict[key] in test_classes:
            test_class_names.append(key)
        if label_dict[key] in train_classes:
            train_class_names.append(key)
        if label_dict[key] in val_classes:
            val_class_names.append(key)
    return train_classes, val_classes, test_classes, train_class_names, val_class_names, test_class_names


def _load_banking77_categories_json(path):
    with open(path, 'r', errors='ignor') as infile:
        categories = []
        all_data = json.load(infile)
        for line in all_data:
            categories.append(line)

    return categories


def _get_banking77_classes(args):
    all_class_randm_idx = np.random.permutation(list(range(77)))
    train_classes = all_class_randm_idx[:30]
    val_classes = all_class_randm_idx[30:45]
    test_classes = all_class_randm_idx[45:]
    print("train classes: ", train_classes)
    print("val classes: ", val_classes)
    print("test classes: ", test_classes)
    class_label = _load_banking77_categories_json(os.path.join(args.data_path, 'categories.json'))
    train_class_names, val_class_names, test_class_names = [], [], []
    for i in train_classes:
        train_class_names.append(class_label[i])
    for i in val_classes:
        val_class_names.append(class_label[i])
    for i in test_classes:
        train_class_names.append(class_label[i])

    return train_classes, val_classes, test_classes, train_class_names, val_class_names, test_class_names


def _get_clinc150_classes(args):
    all_labels = []
    # [3 2 4 7 0][1]
    banking = ["transfer", "transactions", "balance", "freeze_account", "pay_bill",
               "bill_balance", "bill_due", "interest_rate", "routing", "min_payment",
               "order_checks", "pin_change", "report_fraud", "account_blocked", "spending_history"]
    credit_cards = ["credit_score", "report_lost_card", "credit_limit", "rewards_balance", "new_card",
                    "application_status", "card_declined", "international_fees", "apr", "redeem_rewards",
                    "credit_limit_change", "damaged_card", "replacement_card_duration", "improve_credit_score",
                    "expiration_date"]
    kitchen_dining = ["recipe", "restaurant_reviews", "calories", "nutrition_info", "restaurant_suggestion",
                      "ingredients_list", "ingredient_substitution", "cook_time", "food_last", "meal_suggestion",
                      "restaurant_reservation", "confirm_reservation", "how_busy", "cancel_reservation",
                      "accept_reservations"]
    home = ["shopping_list", "shopping_list_update", "next_song", "play_music", "update_playlist",
            "todo_list", "todo_list_update", "calendar", "calendar_update", "what_song",
            "order", "order_status", "reminder", "reminder_update", "smart_home"]
    auto_commute = ["traffic", "directions", "gas", "gas_type", "distance",
                    "current_location", "mpg", "oil_change_when", "oil_change_how", "jump_start",
                    "uber", "schedule_maintenance", "last_maintenance", "tire_pressure", "tire_change"]
    travel = ["book_flight", "book_hotel", "car_rental", "travel_suggestion", "travel_alert",
              "travel_notification", "carry_on", "timezone", "vaccines", "translate",
              "flight_status", "international_visa", "lost_luggage", "plug_type", "exchange_rate"]
    utility = ["time", "alarm", "share_location", "find_phone", "weather",
               "text", "spelling", "make_call", "timer", "date",
               "calculator", "measurement_conversion", "flip_coin", "roll_dice", "definition"]
    work = ["direct_deposit", "pto_request", "taxes", "payday", "w2",
            "pto_balance", "pto_request_status", "next_holiday", "insurance", "insurance_change",
            "schedule_meeting", "pto_used", "meeting_schedule", "rollover_401k", "income"]
    small_talk = ["greeting", "goodbye", "tell_joke", "where_are_you_from", "how_old_are_you",
                  "what_is_your_name", "who_made_you", "thank_you", "what_can_i_ask_you", "what_are_your_hobbies",
                  "do_you_have_pets", "are_you_a_bot", "meaning_of_life", "who_do_you_work_for", "fun_fact"]
    meta = ["change_ai_name", "change_user_name", "cancel", "user_name", "reset_settings",
            "whisper_mode", "repeat", "no", "yes", "maybe",
            "change_language", "change_accent", "change_volume", "change_speed", "sync_device"]
    all_labels.extend(banking)
    all_labels.extend(credit_cards)
    all_labels.extend(kitchen_dining)
    all_labels.extend(home)
    all_labels.extend(auto_commute)
    all_labels.extend(travel)
    all_labels.extend(utility)
    all_labels.extend(work)
    all_labels.extend(small_talk)
    all_labels.extend(meta)
    class_label = all_labels
    dict_domain ={0: banking,
                  1: credit_cards,
                  2: kitchen_dining,
                  3: home,
                  4: auto_commute,
                  5: travel,
                  6: utility,
                  7: work,
                  8: small_talk,
                  9: meta
                  }
    
    
    train_class_names, val_class_names, test_class_names = [], [], []
    if args.cross_domain:
        # train_classes = list(range(15))
        # val_classes = list(range(15))
        # test_classes = list(range(15))
        train_domains, val_domains, test_domains = _get_clinc150_domains(args)
        label_dict = get_label_dict(args)
        for d in train_domains:
            train_class_names.extend(dict_domain[d])
        for d in val_domains:
            val_class_names.extend(dict_domain[d])
        for d in test_domains:
            test_class_names.extend(dict_domain[d])
        # train_classes = list()
        train_classes = []
        test_classes = []
        val_classes = []
        for i in train_class_names:
            train_classes.append(label_dict[i])
        for i in val_class_names:
            val_classes.append(label_dict[i])
        for i in test_class_names:
            test_classes.append(label_dict[i])
        print("train classes: ", train_classes)
        print("val classes: ", val_classes)
        print("test classes: ", test_classes)
        # 跨域任务最后写吧，有点麻烦
    else:
        all_class_randm_idx = np.random.permutation(list(range(150)))
        train_classes = all_class_randm_idx[:60]
        val_classes = all_class_randm_idx[60:75]
        test_classes = all_class_randm_idx[75:]
        
        for i in train_classes:
            train_class_names.append(class_label[i])
        for i in val_classes:
            val_class_names.append(class_label[i])
        for i in test_classes:
            test_class_names.append(class_label[i])

    return train_classes, val_classes, test_classes, train_class_names, val_class_names, test_class_names


def _get_clinc150_domains(args):
    if args.cross_domain:
        all_domain_randm_idx = np.random.permutation(list(range(10)))
        train_domains = all_domain_randm_idx[:4]
        val_domains = all_domain_randm_idx[4:5]
        test_domains = all_domain_randm_idx[5:]
    else:
        train_domains = [0]
        val_domains = [0]
        test_domains = [0]

    return train_domains, val_domains, test_domains


def _load_json(path):
    '''
        20news  amazon reuters
        load data file
        @param path: str, path to the data file
        @return data: list of examples
    '''
    label = {}
    text_len = []
    with open(path, 'r', errors='ignore') as f:
        data = []
        for line in f:
            row = json.loads(line)

            # count the number of examples per label
            if int(row['label']) not in label:
                label[int(row['label'])] = 1
            else:
                label[int(row['label'])] += 1

            item = {
                'label': row['label'],
                # 'text': row['text'][:500],  # truncate the text to 500 tokens
                'raw': _conect_words(row['text'][:500]) 
            }

            text_len.append(len(row['text']))

            data.append(item)

        tprint('Class balance:')
        tprint(label)
        tprint('Avg len: {}'.format(sum(text_len) / (len(text_len))))

        return data, label

def _load_json2(path, args):
    '''
        Liu hwu64
        load data file
        @param path: str, path to the data file
        @return data: list of examples
    '''
    label = {}
    text_len = []
    label_dict = get_label_dict(args)
    with open(path, 'r', errors='ignore') as f:
        data = []
        for line in f:
            row = json.loads(line)
            # count the number of examples per label
            if  args.dataset == '20newsgroup2' or  args.dataset == 'amazon2':
                tmplabel = int(row['label'])
            else:
                tmplabel = int(label_dict[row['label']])
            if tmplabel not in label:
                label[tmplabel] = 1
            else:
                label[tmplabel] += 1

            item = {
                'label': tmplabel,
                # 'text': row['text'][:500],  # truncate the text to 500 tokens
                'raw': row['sentence']
            }

            text_len.append(len(row['sentence'].split(' ')))

            data.append(item)


        tprint('Class balance:')
        tprint(label)
        tprint('Avg len: {}'.format(sum(text_len) / (len(text_len))))

        return data, label



def _conect_words(data):
    '''
        Count the occurrences of all words
        @param data: list of examples
        @return words: list of words (with duplicates)
    '''
    words = ""
    for example in data:
        if example not in [":", "(" ,")"]:
            words = words + " " + example 
        else:
            words = words + example
    return words


def _load_huffpost_json(data_path):
    label = {}
    text_len = []
    with open(data_path, 'r', errors='ignore') as f:
        data = []
        for line in f:
            row = json.loads(line)

            # count the number of examples per label
            if int(row['label']) not in label:
                label[int(row['label'])] = 1
            else:
                label[int(row['label'])] += 1

            item = {
                'label': row['label'],
                # 'text': row['text'][:500],  # truncate the text to 500 tokens
                'raw': _conect_words(row['text']) 
            }

            text_len.append(len(row['text']))

            data.append(item)

        tprint('Class balance:')
        tprint(label)
        tprint('Avg len: {}'.format(sum(text_len) / (len(text_len))))

        return data






def _load_clinc150_data_json(args, path):
    # max_text_limit = args.max_text_len_limits
    # max_text_len = 0
    all_data = []
    all_labels = []
    # [3 2 4 7 0][1]
    banking = ["transfer", "transactions", "balance", "freeze_account", "pay_bill",
               "bill_balance", "bill_due", "interest_rate", "routing", "min_payment",
               "order_checks", "pin_change", "report_fraud", "account_blocked", "spending_history"]
    credit_cards = ["credit_score", "report_lost_card", "credit_limit", "rewards_balance", "new_card",
                    "application_status", "card_declined", "international_fees", "apr", "redeem_rewards",
                    "credit_limit_change", "damaged_card", "replacement_card_duration", "improve_credit_score",
                    "expiration_date"]
    kitchen_dining = ["recipe", "restaurant_reviews", "calories", "nutrition_info", "restaurant_suggestion",
                      "ingredients_list", "ingredient_substitution", "cook_time", "food_last", "meal_suggestion",
                      "restaurant_reservation", "confirm_reservation", "how_busy", "cancel_reservation",
                      "accept_reservations"]
    home = ["shopping_list", "shopping_list_update", "next_song", "play_music", "update_playlist",
            "todo_list", "todo_list_update", "calendar", "calendar_update", "what_song",
            "order", "order_status", "reminder", "reminder_update", "smart_home"]
    auto_commute = ["traffic", "directions", "gas", "gas_type", "distance",
                    "current_location", "mpg", "oil_change_when", "oil_change_how", "jump_start",
                    "uber", "schedule_maintenance", "last_maintenance", "tire_pressure", "tire_change"]
    travel = ["book_flight", "book_hotel", "car_rental", "travel_suggestion", "travel_alert",
              "travel_notification", "carry_on", "timezone", "vaccines", "translate",
              "flight_status", "international_visa", "lost_luggage", "plug_type", "exchange_rate"]
    utility = ["time", "alarm", "share_location", "find_phone", "weather",
               "text", "spelling", "make_call", "timer", "date",
               "calculator", "measurement_conversion", "flip_coin", "roll_dice", "definition"]
    work = ["direct_deposit", "pto_request", "taxes", "payday", "w2",
            "pto_balance", "pto_request_status", "next_holiday", "insurance", "insurance_change",
            "schedule_meeting", "pto_used", "meeting_schedule", "rollover_401k", "income"]
    small_talk = ["greeting", "goodbye", "tell_joke", "where_are_you_from", "how_old_are_you",
                  "what_is_your_name", "who_made_you", "thank_you", "what_can_i_ask_you", "what_are_your_hobbies",
                  "do_you_have_pets", "are_you_a_bot", "meaning_of_life", "who_do_you_work_for", "fun_fact"]
    meta = ["change_ai_name", "change_user_name", "cancel", "user_name", "reset_settings",
            "whisper_mode", "repeat", "no", "yes", "maybe",
            "change_language", "change_accent", "change_volume", "change_speed", "sync_device"]

    all_labels.extend(banking)
    all_labels.extend(credit_cards)
    all_labels.extend(kitchen_dining)
    all_labels.extend(home)
    all_labels.extend(auto_commute)
    all_labels.extend(travel)
    all_labels.extend(utility)
    all_labels.extend(work)
    all_labels.extend(small_talk)
    all_labels.extend(meta)
    label_dict = get_label_dict(args)
    with open(path, 'r', errors='ignor') as infile:
        all_data = json.load(infile)
        domain_list = []
        new_data = []
        for d in all_data:
            for _, data in enumerate(all_data[d]):
                if d == "test" or d == "val" or d == "train":
                    if data[1] in banking:
                        domain_list.append("banking")
                        domain_id = 0
                    elif data[1] in credit_cards:
                        domain_list.append("credit cards")
                        domain_id = 1
                    elif data[1] in kitchen_dining:
                        domain_list.append("kitchen dining")
                        domain_id = 2
                    elif data[1] in home:
                        domain_list.append("home")
                        domain_id = 3
                    elif data[1] in auto_commute:
                        domain_list.append("auto commute")
                        domain_id = 4
                    elif data[1] in travel:
                        domain_list.append("travel")
                        domain_id = 5
                    elif data[1] in utility:
                        domain_list.append("utility")
                        domain_id = 6
                    elif data[1] in work:
                        domain_list.append("work")
                        domain_id = 7
                    elif data[1] in small_talk:
                        domain_list.append("small talk")
                        domain_id = 8
                    elif data[1] in meta:
                        domain_list.append("meta")
                        domain_id = 9
                    else:
                        print(data[1])
                    # if len(text_token_id) > max_text_limit:
                    #     text_token_id = text_token_id[:max_text_limit]
                    #     text_token_id.append(102)
                    # label_raw = data[1].split("_")
                    item = {
                        'label': label_dict[data[1]],
                        # 'text': row['text'][:500],  # truncate the text to 500 tokens
                        'raw': data[0],
                        'domain': domain_id
                    }
                    new_data.append(item)
    return new_data


def _load_data_to_alldata(data_dir, args):
    all_data = []
    if args.dataset == 'banking77':
        categories = _load_banking77_categories_json(os.path.join(data_dir, 'categories.json'))
        data_path1 = os.path.join(data_dir, 'train.csv')
        data_path2 = os.path.join(data_dir, 'test.csv')
        all_data1 = pd.read_csv(data_path1, delimiter=",")
        all_data2 = pd.read_csv(data_path2, delimiter=",")
        all_data = []
        for i in range(len(categories)):
            for j in range(len(all_data1)):
                if all_data1['category'][j] == categories[i]:
                    label = i
                    item = {
                        'label': label,
                        'raw': all_data1['text'][j]
                    }
                    all_data.append(item)
            for j in range(len(all_data2)):
                if all_data2['category'][j] == categories[i]:
                    label = i
                    item = {
                        'label': label,
                        'raw': all_data2['text'][j]
                    }
                    all_data.append(item)
    elif args.dataset == 'clinc150':
        all_data = _load_clinc150_data_json(args, data_dir)
        # return all_data
    return all_data


def get_dataset(train_classes, test_classes, val_classes, data, count, args):
    train_dataset = []
    test_dataset = []
    val_dataset = []

    text2num_dict = get_label_dict(args=args)
    label_dict = {}
    for key in text2num_dict.keys():
        label_dict[text2num_dict[key]] = key
    for d in data:
        if d['label'] in train_classes:
            train_dataset.append(InputExample(
                guid=count,
                text_a=d['raw'],
                label=d['label'],
                text_b=label_dict[d['label']])
            )
        elif d['label'] in test_classes:
            test_dataset.append(InputExample(
                guid=count,
                text_a=d['raw'],
                label=d['label'],
                text_b=label_dict[d['label']])
            )
        elif d['label'] in val_classes:
            val_dataset.append(InputExample(
                guid=count,
                text_a=d['raw'],
                label=d['label'],
                text_b=label_dict[d['label']])
            )
        count += 1
    # import pdb
    # pdb.set_trace()
    return train_dataset, test_dataset, val_dataset


def load_data(args):
    if args.dataset == '20newsgroup' or args.dataset == '20newsgroup2':
        train_classes, val_classes, test_classes, train_class_names, \
        val_class_names, test_class_names = _get_20newsgroup_classes(args)
        args.template = "This is a [MASK] news: [sentence]"
    elif args.dataset == 'amazon' or args.dataset == 'amazon2':
        train_classes, val_classes, test_classes, train_class_names, \
        val_class_names, test_class_names = _get_amazon_classes(args)
        args.template = "This is a [MASK] review: [sentence]"
    elif args.dataset == 'huffpost':
        train_classes, val_classes, test_classes, train_class_names, \
        val_class_names, test_class_names = _get_huffpost_classes(args)
        args.template = "This is a [MASK] news: [sentence]"

    elif args.dataset == 'banking77':
        train_classes, val_classes, test_classes, train_class_names, \
        val_class_names, test_class_names = _get_banking77_classes(args)
        args.template = "This is a [MASK] intent: [sentence]"

    elif args.dataset == 'clinc150':
        train_classes, val_classes, test_classes, train_class_names, \
        val_class_names, test_class_names = _get_clinc150_classes(args)
        args.template = "This is a [MASK] intent: [sentence]"

    elif args.dataset == 'reuters':
        train_classes, val_classes, test_classes, train_class_names, \
        val_class_names, test_class_names = _get_reuters_classes(args)
        args.template = "This is a [MASK] news: [sentence]"

    elif args.dataset == 'hwu64':
        train_classes, val_classes, test_classes, train_class_names, \
        val_class_names, test_class_names = _get_hwu64_classes(args)
        args.template = "This is a [MASK] intent: [sentence]"
    elif args.dataset == 'liu':
        train_classes, val_classes, test_classes, train_class_names, \
        val_class_names, test_class_names = _get_liu_classes(args)
        args.template = "This is a [MASK] intent: [sentence]"

    else:
        raise ValueError(
            'args.dataset should be one of'
            '[20newsgroup, amazon, huffpost, banking77, clinc150]')
    # import pdb
    # pdb.set_trace()
    assert (len(train_classes) == args.n_train_class)
    assert (len(val_classes) == args.n_val_class)
    assert (len(test_classes) == args.n_test_class)

    if args.dataset == "clinc150":
        train_domains, val_domains, test_domains = _get_clinc150_domains(args)
    else:
        train_domains = [0]
        val_domains = [0]
        test_domains = [0]

    args.train_classes = train_classes
    args.val_classes = val_classes
    args.test_classes = test_classes
    args.num_classes = args.n_train_class + args.n_val_class + args.n_test_class
    args.num_domain = 1 if args.n_train_domain == 1 else args.n_train_domain + args.n_val_domain + args.n_test_domain
    args.train_domains = train_domains
    args.val_domains = val_domains
    args.test_domains = test_domains
    print("train_domains: ", train_domains)
    print("val_domains: ", val_domains)
    print("test_domains: ", test_domains)

    tprint('Loading data')

    if args.dataset == "20newsgroup" or args.dataset == 'amazon' or args.dataset == 'reuters':
        all_data, _ = _load_json(args.data_path)
    elif args.dataset == "huffpost":
        all_data = _load_huffpost_json(args.data_path)
    elif args.dataset == "hwu64" or args.dataset == "liu" or args.dataset == 'amazon2' or args.dataset == '20newsgroup2':
        all_data, _ = _load_json2(args.data_path, args)
    else: 
        all_data = _load_data_to_alldata(args.data_path, args)
    train_data, test_data, val_data = get_dataset(train_classes, test_classes, val_classes, all_data, 0, args)
    return train_data, val_data, test_data