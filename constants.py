"""
Constants.
"""
FEATURES = ['neutral_transaction_count', 'SUM(session.MAX(transaction.length))',
       'MAX(transaction.itemInSession)', 'MEAN(transaction.length)',
       'COUNT(transaction WHERE page = Home)',
       'COUNT(transaction WHERE page = Settings)',
       'MEAN(session.MAX(transaction.length))',
       'COUNT(transaction WHERE page = Add to Playlist)',
       'SUM(session.MAX(transaction.itemInSession))',
       'NUM_UNIQUE(transaction.artist)',
       'MAX(session.SUM(transaction.length))', 'COUNT(transaction)',
       'MEAN(session.NUM_UNIQUE(transaction.artist))',
       'MIN(session.MAX(transaction.itemInSession))',
       'MIN(session.SUM(transaction.length))',
       'COUNT(transaction WHERE page = NextSong)',
       'positive_transaction_count', 'MEAN(session.SUM(transaction.length))',
       'COUNT(transaction WHERE page = Logout)', 'SUM(transaction.length)',
       'COUNT(transaction WHERE page = Downgrade)',
       'MEAN(session.NUM_UNIQUE(transaction.song))',
       'MIN(session.COUNT(transaction))', 'NUM_UNIQUE(transaction.song)',
       'MEAN(session.MAX(transaction.itemInSession))',
       'MAX(session.NUM_UNIQUE(transaction.song))',
       'MAX(session.NUM_UNIQUE(transaction.artist))',
       'negative_transaction_count',
       'COUNT(transaction WHERE page = Thumbs Up)',
       'COUNT(transaction WHERE page = Roll Advert)',
       'MEAN(session.MEAN(transaction.length))',
       'MIN(session.MEAN(transaction.length))',
       'MIN(session.NUM_UNIQUE(transaction.song))', 'MAX(transaction.length)',
       'COUNT(transaction WHERE page = Thumbs Down)',
       'MIN(session.NUM_UNIQUE(transaction.artist))',
       'MEAN(session.COUNT(transaction))',
       'MEAN(session.NUM_UNIQUE(transaction.page))',
       'SUM(session.MEAN(transaction.length))',
       'MIN(session.MAX(transaction.length))',
       'SUM(session.NUM_UNIQUE(transaction.artist))',
       'MAX(session.COUNT(transaction))',
       'COUNT(transaction WHERE page = Add Friend)',
       'NUM_UNIQUE(transaction.page)',
       'gender_M']

TARGET = "is_cancel"

CONFIG_FAI = {
    'gender_M': {
        'unprivileged_attribute_values': [0],
        'privileged_attribute_values': [1],
    },
}