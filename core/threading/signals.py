from PySide6.QtCore import Signal, QObject, QPoint

from core.catalog.Coin import Coin


class ThreadingSignals(QObject):

    frame_available = Signal(object)

    s_save_picture_button_pressed = Signal()

    # passing camera index
    camera_reinit_signal = Signal(int)

    new_coin_created = Signal(dict)

    set_coins_combo_box = Signal(list)

    info_signal = Signal(bool, object)

    s_append_info_text = Signal(str)

    s_catalog_changed = Signal()

    s_active_coin_changed = Signal(Coin)

    s_active_tab_changed = Signal()

    s_coin_photo_id_changed = Signal(int)

    s_coin_vertices_update = Signal(list)

    s_reset_vertices = Signal()
