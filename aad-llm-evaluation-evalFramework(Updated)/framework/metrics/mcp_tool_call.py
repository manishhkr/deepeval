class MCPToolCall:
    def __init__(self, name, args=None, result=None):
        self.name = name
        self.args = args or {}
        self.result = result or {}

    def to_dict(self):
        return {
            "name": self.name,
            "args": self.args,
            "result": self.result
        }
