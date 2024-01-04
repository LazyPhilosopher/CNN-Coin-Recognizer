from PySide6.QtCore import Signal, QObject


class ThreadingSignals(QObject):

    frame_available = Signal(object)

    save_picture = Signal(object)

    # passing camera index
    camera_reinit_signal = Signal(int)

    new_coin_created = Signal(dict)

    set_coins_combo_box = Signal(list)

    info_signal = Signal(bool, object)

    s_append_info_text = Signal(str)

    s_catalog_changed = Signal()
