class PythonSyntaxHighlighter(QSyntaxHighlighter):
    def __init__(self, document):
        super().__init__(document)
        
        self.highlighting_rules = []
        
        # Keywords
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor(120, 120, 250))
        keyword_format.setFontWeight(QFont.Weight.Bold)
        keywords = [
            r'\bimport\b', r'\bfrom\b', r'\bdef\b', r'\bclass\b', r'\bfor\b', r'\bwhile\b',
            r'\bif\b', r'\belif\b', r'\belse\b', r'\btry\b', r'\bexcept\b', r'\bfinally\b',
            r'\breturn\b', r'\bpass\b', r'\bcontinue\b', r'\bbreak\b', r'\bin\b', r'\bas\b',
            r'\bglobal\b', r'\bwith\b', r'\braise\b', r'\bTrue\b', r'\bFalse\b', r'\bNone\b'
        ]
        for pattern in keywords:
            regex = QRegularExpression(pattern)
            self.highlighting_rules.append((regex, keyword_format))
        
        # Functions
        function_format = QTextCharFormat()
        function_format.setForeground(QColor(40, 170, 40))
        function_regex = QRegularExpression(r'\b[A-Za-z0-9_]+(?=\()')
        self.highlighting_rules.append((function_regex, function_format))
        
        # String literals
        string_format = QTextCharFormat()
        string_format.setForeground(QColor(220, 120, 70))
        string_regex1 = QRegularExpression(r'"[^"\\]*(\\.[^"\\]*)*"')
        string_regex2 = QRegularExpression(r"'[^'\\]*(\\.[^'\\]*)*'")
        self.highlighting_rules.append((string_regex1, string_format))
        self.highlighting_rules.append((string_regex2, string_format))
        
        # Comments
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor(120, 120, 120))
        comment_format.setFontItalic(True)
        comment_regex = QRegularExpression(r'#[^\n]*')
        self.highlighting_rules.append((comment_regex, comment_format))
        
    def highlightBlock(self, text):
        for regex, format in self.highlighting_rules:
            match_iterator = regex.globalMatch(text)
            while match_iterator.hasNext():
                match = match_iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), format)


class CustomPlotDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Custom Matplotlib Plot")
        self.setMinimumSize(800, 600)
        
        self.setup_ui()
        
        # Default sample code
        sample_code = """# Sample code for a bar graph from cell_data.csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = pd.read_csv('/Users/clark/Desktop/wang/protein_visualization_app/assets/sample_data/cell_data.csv')

# Create figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Select first 5 rows and some protein columns for the bar chart
sample_data = data.iloc[:5, 3:8]

# Transpose so proteins become the index
sample_data = sample_data.transpose()

# Create the bar chart
sample_data.plot(kind='bar', ax=ax)

# Customize the plot
ax.set_title('Protein Expression Levels in First 5 Cells')
ax.set_xlabel('Proteins')
ax.set_ylabel('Expression Level')
ax.legend(title='Cell ID', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout
plt.tight_layout()

# Return the figure for display
fig  # This figure will be displayed
"""
        
        self.code_edit.setPlainText(sample_code)
        
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        # Code editor
        code_group = QGroupBox("Python Code:")
        code_layout = QVBoxLayout()
        
        self.code_edit = QPlainTextEdit()
        self.code_edit.setFont(QFont("Courier New", 10))
        self.code_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        code_layout.addWidget(self.code_edit)
        
        # Setup syntax highlighter
        self.highlighter = PythonSyntaxHighlighter(self.code_edit.document())
        
        code_group.setLayout(code_layout)
        
        # Output area
        output_group = QGroupBox("Output:")
        output_layout = QVBoxLayout()
        
        self.output_text = QPlainTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Courier New", 10))
        self.output_text.setMaximumHeight(100)
        output_layout.addWidget(self.output_text)
        
        output_group.setLayout(output_layout)
        
        # Plot area
        plot_group = QGroupBox("Plot Preview:")
        plot_layout = QVBoxLayout()
        
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        
        plot_group.setLayout(plot_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.insert_button)
        button_layout.addWidget(self.cancel_button)
        
        # Add all components to main layout
        main_layout.addWidget(code_group, 3)
        main_layout.addWidget(output_group, 1)
        main_layout.addWidget(plot_group, 3)
        main_layout.addLayout(button_layout)
       