from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QStyleFactory,
    QTabWidget,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)
from PyQt5.QtGui import QIcon
from lidarsegmentation.settings.coord_settings import CS
from lidarsegmentation.settings.seg_settings import SS
import os
from lidarsegmentation.coordinates import coordinates
from lidarsegmentation.merge_coordinates import merge_coordinates
from lidarsegmentation.clear_excess_stumps import clear_excess_stumps
import ast
import pandas as pd
from typing import List, Dict, Tuple, Any

from lidarsegmentation.segmentation_vor import segmentation_vor
from lidarsegmentation.segmentation_ram import segmentation_ram
from lidarsegmentation.segmentation_clear import segmentation_clear

# from seg_after import seg_after
# from orbit_gif import orbit_gif
from lidarsegmentation.predict import predict
from lidarsegmentation.parameters import parameters


def str_to_bool(s: str) -> bool:
    return s == 'True'


class WidgetGallery(QWidget):
    def __init__(self, parent=None):
        super(WidgetGallery, self).__init__(parent)

        self.setWindowIcon(QIcon(os.path.join('lidarsegmentation', 'logo', 'logo.png')))

        self.resize(750, 750)

        self.file_path_TRAJ = 'empty'
        self.file_path_SHAPE = 'empty'
        self.file_path_COORD = 'empty'
        self.file_path_SETTINGS = os.path.join('lidarsegmentation', 'settings', 'settings.yaml')

        self.createTopLeftGroupBox()
        self.createRightTabWidget()
        self.createTopRightGroupBox()
        self.createBottomRightGroupBox()
        self.createBottomLeftGroupBox()

        # styleLabel = QLabel("Style:")

        topLayout = QHBoxLayout()
        topLayout.addStretch(1)
        # topLayout.addWidget(styleLabel)

        mainLayout = QGridLayout()
        mainLayout.addLayout(topLayout, 0, 0, 1, 2)
        mainLayout.addWidget(self.topLeftGroupBox, 1, 0)
        mainLayout.addWidget(self.topRightGroupBox, 1, 1)
        mainLayout.addWidget(self.bottomLeftGroupBox, 2, 0)
        mainLayout.addWidget(self.bottomRightGroupBox, 2, 1)

        mainLayout.addWidget(self.rightTabWidget, 1, 3, 2, 1)

        mainLayout.setRowStretch(1, 1)
        mainLayout.setRowStretch(2, 1)
        mainLayout.setColumnStretch(0, 1)
        mainLayout.setColumnStretch(1, 1)
        self.setLayout(mainLayout)

        self.setWindowTitle('pcPCD')
        self.changeStyle('Fusion')

    def changeStyle(self, styleName: str) -> None:
        QApplication.setStyle(QStyleFactory.create(styleName))

    def advanceProgressBar(self) -> None:
        curVal = self.progressBar.value()
        maxVal = self.progressBar.maximum()
        self.progressBar.setValue(curVal + (maxVal - curVal) // 100)

    def createTopLeftGroupBox(self) -> None:
        self.topLeftGroupBox = QGroupBox('Coordinates')
        self.coordinate_thresholds: List[int] = [7000, 5000, 1000]
        self.coord_checkboxes: List[QCheckBox] = []
        layout = QVBoxLayout()
        for threshold in self.coordinate_thresholds:
            checkbox = QCheckBox(f'Coordinates (int = {threshold})')
            self.coord_checkboxes.append(checkbox)
            layout.addWidget(checkbox)
        self.merge_coords_checkbox = QCheckBox('Merge Coordinates')
        self.clear_excess_checkbox = QCheckBox('Clear Excess Stumps')
        self.clear_excess_checkbox.stateChanged.connect(self.auto_file_coords)
        layout.addWidget(self.merge_coords_checkbox)
        layout.addWidget(self.clear_excess_checkbox)
        layout.addStretch(1)
        self.topLeftGroupBox.setLayout(layout)

    def createBottomLeftGroupBox(self) -> None:
        self.bottomLeftGroupBox = QGroupBox('Segmentation and Parameters')
        seg_labels: List[str] = [
            'Segmentation Voronoi',
            'Segmentation RAM',
            'Segmentation Clear',
            'Predict Labels',
            'Estimate Parameters',
        ]
        self.seg_checkboxes: List[QCheckBox] = []
        layout = QVBoxLayout()
        for label in seg_labels:
            checkbox = QCheckBox(label)
            checkbox.stateChanged.connect(self.seg_file_coords)
            self.seg_checkboxes.append(checkbox)
            layout.addWidget(checkbox)
        layout.addStretch(1)
        self.bottomLeftGroupBox.setLayout(layout)

    def createTopRightGroupBox(self):
        self.topRightGroupBox = QGroupBox('Control')

        self.disableWidgetsCheckBox = QCheckBox('Выбрать настройки из файла *.yaml')

        self.button0 = QPushButton('* Файл настроек', self)
        self.button0.clicked.connect(self.open_file_dialog_SETTINGS)
        self.file_path_label0 = QLabel(' ', self)
        self.button0.setDisabled(True)
        self.file_path_label0.setDisabled(True)

        self.startButton = QPushButton('НАЧАТЬ ОБРАБОТКУ')
        self.startButton.clicked.connect(self.start)

        self.labelempty000 = QLabel('')
        self.labelempty001 = QLabel('')

        self.selectFolderBtn = QPushButton('* Папка проекта', self)
        self.selectFolderBtn.clicked.connect(self.selectFolder)
        self.selectedPathLabel = QLabel(' ', self)

        self.button1 = QPushButton('* Файл с данными облака', self)
        self.button1.clicked.connect(self.open_file_dialog_POINTS)
        self.file_path_label1 = QLabel(' ', self)

        self.button2 = QPushButton('* Файл с треком человека', self)
        self.button2.clicked.connect(self.open_file_dialog_TRAJ)
        self.file_path_label2 = QLabel(' ', self)

        self.button3 = QPushButton('Файл с границами участка', self)
        self.button3.clicked.connect(self.open_file_dialog_SHAPE)
        self.file_path_label3 = QLabel('Опционально', self)

        self.button4 = QPushButton('* Файл с координатами', self)
        self.button4.clicked.connect(self.open_file_dialog_COORD)
        self.file_path_label4 = QLabel(' ', self)

        self.disableWidgetsCheckBox.toggled.connect(self.rightTabWidget.setDisabled)

        self.disableWidgetsCheckBox.toggled.connect(self.selectFolderBtn.setDisabled)
        self.disableWidgetsCheckBox.toggled.connect(self.selectedPathLabel.setDisabled)

        self.disableWidgetsCheckBox.toggled.connect(self.button0.setEnabled)
        self.disableWidgetsCheckBox.toggled.connect(self.button0.setEnabled)

        self.disableWidgetsCheckBox.toggled.connect(self.button1.setDisabled)
        self.disableWidgetsCheckBox.toggled.connect(self.file_path_label1.setDisabled)

        self.disableWidgetsCheckBox.toggled.connect(self.button1.setDisabled)
        self.disableWidgetsCheckBox.toggled.connect(self.file_path_label1.setDisabled)

        self.disableWidgetsCheckBox.toggled.connect(self.button2.setDisabled)
        self.disableWidgetsCheckBox.toggled.connect(self.file_path_label2.setDisabled)

        self.disableWidgetsCheckBox.toggled.connect(self.button3.setDisabled)
        self.disableWidgetsCheckBox.toggled.connect(self.file_path_label3.setDisabled)

        self.disableWidgetsCheckBox.toggled.connect(self.button4.setDisabled)
        self.disableWidgetsCheckBox.toggled.connect(self.file_path_label4.setDisabled)

        self.button2.hide()
        self.file_path_label2.hide()

        self.button4.hide()
        self.file_path_label4.hide()

        layout = QVBoxLayout()
        layout.addWidget(self.disableWidgetsCheckBox)
        layout.addWidget(self.button0)
        layout.addWidget(self.file_path_label0)
        layout.addWidget(self.selectFolderBtn)
        layout.addWidget(self.selectedPathLabel)
        layout.addWidget(self.button1)
        layout.addWidget(self.file_path_label1)
        layout.addWidget(self.button2)
        layout.addWidget(self.file_path_label2)
        layout.addWidget(self.button3)
        layout.addWidget(self.file_path_label3)
        layout.addWidget(self.button4)
        layout.addWidget(self.file_path_label4)
        layout.addWidget(self.labelempty000)
        layout.addWidget(self.labelempty001)
        layout.addWidget(self.startButton)
        layout.addStretch(1)

        self.topRightGroupBox.setLayout(layout)

    def selectFolder(self):
        self.folderPath = QFileDialog.getExistingDirectory(self, 'Выберите папку проекта', '')
        self.selectedPathLabel.setText(self.folderPath)
        self.selectFolderBtn.setText('Папка проекта')

    def open_file_dialog_POINTS(self):
        options = QFileDialog.Options()
        self.file_path_POINTS, _ = QFileDialog.getOpenFileName(
            self, 'Выберите файл с данными облака', '', 'Point Cloud Files (*.las *.pcd)', options=options
        )
        if self.file_path_POINTS:
            self.file_path_label1.setText(self.file_path_POINTS)
            self.button1.setText('Файл с данными облака')

    def open_file_dialog_SETTINGS(self):
        options = QFileDialog.Options()
        self.file_path_SETTINGS, _ = QFileDialog.getOpenFileName(
            self, 'Выберите файл настроек', '', 'YAML Files (*.yaml *.yml)', options=options
        )
        if self.file_path_SETTINGS:
            self.file_path_label0.setText(self.file_path_SETTINGS)
            self.button0.setText('Файл настроек')

    def open_file_dialog_TRAJ(self):
        options = QFileDialog.Options()
        self.file_path_TRAJ, _ = QFileDialog.getOpenFileName(
            self, 'Выберите файл с треком человека', '', 'Point Cloud Files (*.las *.pcd)', options=options
        )
        if self.file_path_TRAJ:
            self.file_path_label2.setText(self.file_path_TRAJ)
            self.button2.setText('Файл с треком человека')

    def open_file_dialog_SHAPE(self):
        options = QFileDialog.Options()
        self.file_path_SHAPE, _ = QFileDialog.getOpenFileName(
            self, 'Выберите файл с границами участка', '', 'Shapes Files (*.shp)', options=options
        )
        if self.file_path_SHAPE:
            self.file_path_label3.setText(self.file_path_SHAPE)

    def open_file_dialog_COORD(self):
        options = QFileDialog.Options()
        self.file_path_COORD, _ = QFileDialog.getOpenFileName(
            self, 'Выберите файл с координатами', '', 'CSV Files (*.csv)', options=options
        )
        if self.file_path_COORD:
            self.file_path_label4.setText(self.file_path_COORD)
            self.button4.setText('Файл с координатами')

    def seg_file_coords(self) -> None:
        if any(cb.isChecked() for cb in self.seg_checkboxes):
            self.button4.show()
            self.file_path_label4.show()
        else:
            self.button4.hide()
            self.file_path_label4.hide()

    def add_to_grid(self, layout, info, lbl, obj, i):
        layout.addWidget(info, i, 0, 1, 2)
        layout.addWidget(lbl, i + 1, 0)
        layout.addWidget(obj, i + 1, 1)
        i = i + 2
        return layout, i

    def auto_file_coords(self) -> None:
        if self.clear_excess_checkbox.isChecked():
            self.file_path_label4.setText('Файл будет создан автоматически')
        else:
            if self.file_path_COORD != 'empty':
                self.file_path_label4.setText(self.file_path_COORD)
            else:
                self.file_path_label4.setText(' ')

    def createRightTabWidget(self):
        self.rightTabWidget = QTabWidget()
        self.rightTabWidget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Ignored)

        tab1 = QWidget()

        self.cut_data_info = QLabel('Обрезка данных по высоте и границам участка')
        self.cut_data_label = QLabel('FLAG_cut_data')
        self.cut_data_combo = QComboBox(self)
        self.cut_data_combo.addItem('True')
        self.cut_data_combo.addItem('False')
        self.cut_data_combo.setCurrentIndex(0)

        self.cells_info = QLabel('Выделение подобластей')
        self.cells_label = QLabel('FLAG_make_cells')
        self.cells_combo = QComboBox(self)
        self.cells_combo.addItem('True')
        self.cells_combo.addItem('False')
        self.cells_combo.setCurrentIndex(0)

        self.stumps_info = QLabel('Выделение пеньков деревьев и вычисление координат')
        self.stumps_label = QLabel('FLAG_make_stumps')
        self.stumps_combo = QComboBox(self)
        self.stumps_combo.addItem('True')
        self.stumps_combo.addItem('False')
        self.stumps_combo.setCurrentIndex(0)

        self.cells_method_info = QLabel('Метод выделения подобластей')
        self.cells_method_label = QLabel('cut_data_method')
        self.cells_method_combo = QComboBox(self)
        self.cells_method_combo.addItem('voronoi_tessellation')
        self.cells_method_combo.addItem('flood_fill')
        self.cells_method_combo.setCurrentIndex(0)
        self.cells_method_combo.currentTextChanged.connect(self.on_combobox_changed)

        self.low_bound_info = QLabel('Нижняя граница рассматриваемого слоя точек')
        self.low_bound_label = QLabel('LOW')
        self.low_bound_input = QLineEdit(self)
        self.low_bound_input.setText('0.0')

        self.up_bound_info = QLabel('Верхняя граница рассматриваемого слоя точек')
        self.up_bound_label = QLabel('UP')
        self.up_bound_input = QLineEdit(self)
        self.up_bound_input.setText('3.0')

        self.x_shift_info = QLabel('Сдвиг по Х облака точек')
        self.x_shift_label = QLabel('x_shift')
        self.x_shift_input = QLineEdit(self)
        self.x_shift_input.setText('0')

        self.y_shift_info = QLabel('Сдвиг по Y облака точек')
        self.y_shift_label = QLabel('y_shift')
        self.y_shift_input = QLineEdit(self)
        self.y_shift_input.setText('0')

        self.z_shift_info = QLabel('Сдвиг по Z облака точек')
        self.z_shift_label = QLabel('z_shift')
        self.z_shift_input = QLineEdit(self)
        self.z_shift_input.setText('0')

        self.clustering_algo_info = QLabel('Метод кластеризации')
        self.clustering_algo_label = QLabel('algo')
        self.clustering_algo_combo = QComboBox(self)
        self.clustering_algo_combo.addItem('birch')
        self.clustering_algo_combo.addItem('spectral')
        self.clustering_algo_combo.addItem('kmeans')
        self.clustering_algo_combo.setCurrentIndex(0)

        self.n_clusters_info = QLabel('Количество кластеров при разделение участка на подобласти (voronoi_tessellation)')
        self.n_clusters_label = QLabel('n_clusters')
        self.n_clusters_input = QLineEdit(self)
        self.n_clusters_input.setText('32')

        self.cell_size_info = QLabel('Размерность ячейки для выделения ячеек по границам трека человека')
        self.cell_size_label = QLabel('cell_size')
        self.cell_size_input = QLineEdit(self)
        self.cell_size_input.setText('0.20')

        self.height_limit1_info = QLabel('Лимит по минимальной высоте извлеченных пеньков на первом этапе фильтрации')
        self.height_limit1_label = QLabel('height_limit_1')
        self.height_limit1_input = QLineEdit(self)
        self.height_limit1_input.setText('1.25')

        self.height_limit2_info = QLabel('Лимит по минимальной высоте извлеченных пеньков на втором этапе фильтрации')
        self.height_limit2_label = QLabel('height_limit_2')
        self.height_limit2_input = QLineEdit(self)
        self.height_limit2_input.setText('1.35')

        self.eps_xy_info = QLabel('Параметр алгоритма DBSCAN по осям XY')
        self.eps_xy_label = QLabel('eps_XY')
        self.eps_xy_input = QLineEdit(self)
        self.eps_xy_input.setText('0.08')

        self.eps_z_info = QLabel('Параметр алгоритма DBSCAN по оси Z')
        self.eps_z_label = QLabel('eps_Z')
        self.eps_z_input = QLineEdit(self)
        self.eps_z_input.setText('0.7')

        tab1hbox = QGridLayout()
        tab1hbox.setContentsMargins(5, 5, 5, 5)

        i = 1
        tab1hbox, i = self.add_to_grid(tab1hbox, self.cut_data_info, self.cut_data_label, self.cut_data_combo, i)
        tab1hbox, i = self.add_to_grid(tab1hbox, self.cells_info, self.cells_label, self.cells_combo, i)
        tab1hbox, i = self.add_to_grid(tab1hbox, self.stumps_info, self.stumps_label, self.stumps_combo, i)
        tab1hbox, i = self.add_to_grid(tab1hbox, self.cells_method_info, self.cells_method_label, self.cells_method_combo, i)
        tab1hbox, i = self.add_to_grid(tab1hbox, self.low_bound_info, self.low_bound_label, self.low_bound_input, i)
        tab1hbox, i = self.add_to_grid(tab1hbox, self.up_bound_info, self.up_bound_label, self.up_bound_input, i)
        tab1hbox, i = self.add_to_grid(tab1hbox, self.x_shift_info, self.x_shift_label, self.x_shift_input, i)
        tab1hbox, i = self.add_to_grid(tab1hbox, self.y_shift_info, self.y_shift_label, self.y_shift_input, i)
        tab1hbox, i = self.add_to_grid(tab1hbox, self.z_shift_info, self.z_shift_label, self.z_shift_input, i)
        tab1hbox, i = self.add_to_grid(tab1hbox, self.clustering_algo_info, self.clustering_algo_label, self.clustering_algo_combo, i)
        tab1hbox, i = self.add_to_grid(tab1hbox, self.n_clusters_info, self.n_clusters_label, self.n_clusters_input, i)
        tab1hbox, i = self.add_to_grid(tab1hbox, self.cell_size_info, self.cell_size_label, self.cell_size_input, i)
        tab1hbox, i = self.add_to_grid(tab1hbox, self.height_limit1_info, self.height_limit1_label, self.height_limit1_input, i)
        tab1hbox, i = self.add_to_grid(tab1hbox, self.height_limit2_info, self.height_limit2_label, self.height_limit2_input, i)
        tab1hbox, i = self.add_to_grid(tab1hbox, self.eps_xy_info, self.eps_xy_label, self.eps_xy_input, i)
        tab1hbox, i = self.add_to_grid(tab1hbox, self.eps_z_info, self.eps_z_label, self.eps_z_input, i)

        tab1.setLayout(tab1hbox)

        tab2 = QWidget()

        self.first_num_info = QLabel('Номер дерева, с которого начнется извлечение')
        self.first_num_label = QLabel('first_num')
        self.first_num_input = QLineEdit(self)
        self.first_num_input.setText('0')

        self.step_info = QLabel('Шаг просмотро по высоте')
        self.step_label = QLabel('STEP')
        self.step_input = QLineEdit(self)
        self.step_input.setText('2.5')

        self.z_thresholds_info = QLabel('Пороги по высоте, до которых действуют eps_steps и min_pts')
        self.z_thresholds_label = QLabel('z_thresholds')
        self.z_thresholds_input = QLineEdit(self)
        self.z_thresholds_input.setText('[0.5, 0.625, 0.695, 0.75, 0.875, 1]')

        self.eps_steps_info = QLabel('eps алгоритма DBSCAN, которые дествуют при z_thresholds (0.35 + eps_steps[i])')
        self.eps_steps_label = QLabel('eps_steps')
        self.eps_steps_input = QLineEdit(self)
        self.eps_steps_input.setText('[0.01, 0.15, 0.35, 0.5, 0.6, 0.7]')

        self.min_pts_info = QLabel('minPts алгоритма DBSCAN, которые дествуют при z_thresholds')
        self.min_pts_label = QLabel('min_pts')
        self.min_pts_input = QLineEdit(self)
        self.min_pts_input.setText('[50, 50, 50, 50, 45, 40]')

        tab2hbox = QGridLayout()
        tab2hbox.setContentsMargins(5, 5, 5, 5)

        i = 1
        tab2hbox, i = self.add_to_grid(tab2hbox, self.first_num_info, self.first_num_label, self.first_num_input, i)
        tab2hbox, i = self.add_to_grid(tab2hbox, self.step_info, self.step_label, self.step_input, i)
        tab2hbox, i = self.add_to_grid(tab2hbox, self.z_thresholds_info, self.z_thresholds_label, self.z_thresholds_input, i)
        tab2hbox, i = self.add_to_grid(tab2hbox, self.eps_steps_info, self.eps_steps_label, self.eps_steps_input, i)
        tab2hbox, i = self.add_to_grid(tab2hbox, self.min_pts_info, self.min_pts_label, self.min_pts_input, i)

        self.labelempty = QLabel('')
        tab2hbox.addWidget(self.labelempty, i, 0, 1, 2)
        self.labelempty1 = QLabel('')
        tab2hbox.addWidget(self.labelempty1, i + 2, 0, 1, 2)
        self.labelempty2 = QLabel('')
        tab2hbox.addWidget(self.labelempty2, i + 4, 0, 1, 2)
        self.labelempty3 = QLabel('')
        tab2hbox.addWidget(self.labelempty3, i + 6, 0, 1, 2)
        self.labelempty4 = QLabel('')
        tab2hbox.addWidget(self.labelempty4, i + 8, 0, 1, 2)

        self.labelempty5 = QLabel('')
        tab2hbox.addWidget(self.labelempty5, i + 10, 0, 1, 2)
        self.labelempty6 = QLabel('')
        tab2hbox.addWidget(self.labelempty6, i + 12, 0, 1, 2)
        self.labelempty7 = QLabel('')
        tab2hbox.addWidget(self.labelempty7, i + 14, 0, 1, 2)
        self.labelempty8 = QLabel('')
        tab2hbox.addWidget(self.labelempty8, i + 16, 0, 1, 2)
        self.labelempty9 = QLabel('')
        tab2hbox.addWidget(self.labelempty9, i + 18, 0, 1, 2)

        tab2.setLayout(tab2hbox)

        self.rightTabWidget.addTab(tab1, 'Coordinates Settings')
        self.rightTabWidget.addTab(tab2, 'Segmentation Settings')

    def on_combobox_changed(self):
        if self.cells_method_combo.currentText() == 'flood_fill':
            self.button2.show()
            self.file_path_label2.show()
        else:
            self.button2.hide()
            self.file_path_label2.hide()

    def createBottomRightGroupBox(self):
        self.bottomRightGroupBox = QGroupBox('INFO')
        layout = QVBoxLayout()
        self.textEdit = QPlainTextEdit()
        self.textEdit.setReadOnly(True)

        layout.addWidget(self.textEdit)
        self.bottomRightGroupBox.setLayout(layout)

    def start(self):
        if self.disableWidgetsCheckBox.isChecked():
            cs = CS.from_yaml(self.file_path_SETTINGS)
            ss = SS.from_yaml(self.file_path_SETTINGS)
        if not self.disableWidgetsCheckBox.isChecked():
            relative_path_points = os.path.relpath(self.file_path_POINTS, self.folderPath)
            if self.file_path_TRAJ == 'empty':
                relative_path_traj = 'empty.file'
            else:
                relative_path_traj = os.path.relpath(self.file_path_TRAJ, self.folderPath)
            if self.file_path_SHAPE == 'empty':
                relative_path_shape = 'empty.file'
            else:
                relative_path_shape = os.path.relpath(self.file_path_SHAPE, self.folderPath)
            if self.file_path_COORD == 'empty':
                relative_path_coord = 'empty.file'
            else:
                relative_path_coord = os.path.relpath(self.file_path_COORD, self.folderPath)

            if any(cb.isChecked() for cb in self.seg_checkboxes):
                save_pth = relative_path_points.partition('.')[0] + '_Clear_Excess.csv'
                self.file_path_COORD = os.path.join(self.folderPath, save_pth)
                relative_path_coord = os.path.relpath(self.file_path_COORD, self.folderPath)

            cs = CS(
                FLAG_cut_data=str_to_bool(self.cut_data_combo.currentText()),
                FLAG_make_cells=str_to_bool(self.cells_combo.currentText()),
                FLAG_make_stumps=str_to_bool(self.stumps_combo.currentText()),
                cut_data_method=self.cells_method_combo.currentText(),
                LOW=float(self.low_bound_input.text()),
                UP=float(self.up_bound_input.text()),
                x_shift=float(self.x_shift_input.text()),
                y_shift=float(self.y_shift_input.text()),
                z_shift=float(self.z_shift_input.text()),
                algo=self.clustering_algo_combo.currentText(),
                n_clusters=int(self.n_clusters_input.text()),
                cell_size=float(self.cell_size_input.text()),
                height_limit_1=float(self.height_limit1_input.text()),
                height_limit_2=float(self.height_limit2_input.text()),
                eps_XY=float(self.eps_xy_input.text()),
                eps_Z=float(self.eps_z_input.text()),
                path_base=self.folderPath,
                fname_points=relative_path_points,
                fname_traj=relative_path_traj,
                fname_shape=relative_path_shape,
            )

            ss = SS(
                path_base=self.folderPath,
                fname_points=relative_path_points,
                fname_shape=relative_path_shape,
                csv_name_coord=relative_path_coord,
                first_num=int(self.first_num_input.text()),
                STEP=float(self.step_input.text()),
                z_thresholds=ast.literal_eval(self.z_thresholds_input.text()),
                eps_steps=ast.literal_eval(self.eps_steps_input.text()),
                min_pts=ast.literal_eval(self.min_pts_input.text()),
            )

        print(f'main.py {cs.FLAG_make_cells}')

        # In-memory coordinate extraction pipeline
        dfs: List[pd.DataFrame] = []
        pcd_maps: Dict[Any, Any] = {}

        for threshold, checkbox in zip(self.coordinate_thresholds, self.coord_checkboxes):
            if checkbox.isChecked():
                df_chunk, pcd_map_chunk = coordinates(threshold, cs)
                dfs.append(df_chunk)
                pcd_maps.update(pcd_map_chunk)
                self.textEdit.appendPlainText(f'Done processing Coordinates(int = {threshold})')

        # In-memory merge coordinates
        merged_df = pd.DataFrame()
        if self.merge_coords_checkbox.isChecked() and dfs:
            merged_df = merge_coordinates(cs, dfs)
            self.textEdit.appendPlainText('Done processing Merge Coordinates')
        elif dfs:
            # If not merging, just use the last DataFrame
            merged_df = pd.concat(dfs, axis=1)

        # In-memory clear excess stumps
        final_df = pd.DataFrame()
        if self.clear_excess_checkbox.isChecked() and not merged_df.empty:
            final_df = clear_excess_stumps(cs, merged_df, pcd_maps)
            # Write the single output file
            save_pth = cs.fname_points.partition('.')[0] + '_Clear_Excess.csv'
            save_pth = os.path.join(cs.path_base, save_pth)
            final_df.to_csv(save_pth, index=False, sep=';')
            self.textEdit.appendPlainText('Done processing Clear Excess Stumps')
            self.textEdit.appendPlainText(f'Saved final coordinates to {save_pth}')
        elif not merged_df.empty:
            final_df = merged_df
            # Write the merged file if clear_excess was not run
            save_pth = cs.fname_points.partition('.')[0] + '_Coordinates_Merged.csv'
            save_pth = os.path.join(cs.path_base, save_pth)
            final_df.to_csv(save_pth, index=False, sep=';')
            self.textEdit.appendPlainText(f'Saved merged coordinates to {save_pth}')

        # In-memory segmentation pipeline
        seg_voronoi_cb, seg_ram_cb, seg_clear_cb, seg_predict_cb, seg_params_cb = self.seg_checkboxes
        binding_df = None
        vor_trees = {}
        combined_df = None
        ram_trees = {}
        clear_trees = {}
        pred_df = None
        params_df = None
        model_name = 'cpl1-1024-rp-s1024-pn2'

        # Only execute each step if the corresponding checkbox is checked
        if seg_voronoi_cb.isChecked():
            binding_df, vor_trees = segmentation_vor(ss, make_binding=True)
            self.textEdit.appendPlainText(f'Done processing Segmentation Voronoi ({len(vor_trees)} trees)')

        if seg_ram_cb.isChecked() and binding_df is not None and vor_trees:
            combined_df, ram_trees = segmentation_ram(ss, binding_df, vor_trees)
            self.textEdit.appendPlainText(f'Done processing Segmentation RAM ({len(ram_trees)} trees)')

        if seg_clear_cb.isChecked() and combined_df is not None and ram_trees:
            clear_trees = segmentation_clear(ss, combined_df, ram_trees)
            self.textEdit.appendPlainText(f'Done processing Segmentation Clear ({len(clear_trees)} trees)')

            # Save final PCD files if needed
            save_path = os.path.join(ss.path_base, ss.step1_folder_name, ss.step2_folder_name, ss.step3_folder_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for fname, pcd_obj in clear_trees.items():
                pcd_obj.save(os.path.join(save_path, fname))

        if seg_predict_cb.isChecked() and clear_trees:
            pred_df = predict(clear_trees, model_name)
            # Save prediction results
            pred_path = os.path.join(
                ss.path_base, ss.step1_folder_name, ss.step2_folder_name, ss.step3_folder_name, 'predict_' + model_name + '.csv'
            )
            pred_df.to_csv(pred_path, index=False, sep=';')
            self.textEdit.appendPlainText(f'Done processing Predict Labels (saved to {pred_path})')

        if seg_params_cb.isChecked() and combined_df is not None and clear_trees:
            try:
                params_df = parameters(ss, combined_df, clear_trees)
                # Save parameters
                param_name = ss.fname_points.partition('.')[0] + '_Parameters.csv'
                param_path = os.path.join(ss.path_base, param_name)
                params_df.to_csv(param_path, index=False, sep=';')
                self.textEdit.appendPlainText(f'Done processing Estimate Parameters (saved to {param_path})')
            except Exception as e:
                self.textEdit.appendPlainText(f'Error estimating parameters: {e}')
                self.textEdit.appendPlainText('Please check the log for details')

        self.textEdit.appendPlainText('All steps done')


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    gallery = WidgetGallery()
    gallery.show()
    sys.exit(app.exec())
