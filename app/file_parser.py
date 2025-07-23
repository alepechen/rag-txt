import logging

class TxtParser:
    def __init__(self, filepath: str):
        self.filepath = filepath
        
    def parse(self, filepath: str) -> str:
        """Parses a text file and returns its content."""
        try:
            with open(filepath, 'r') as file:
                return file.read()
        except Exception as e:
            logging.error(f"Error reading text file: {e}")
            return "Error reading text file"