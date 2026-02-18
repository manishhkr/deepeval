class MCPServer:
    def __init__(self, name, transport="streamable-http",
                 tools=None, resources=None, prompts=None):
        self.name = name
        self.transport = transport
        self.tools = tools or []
        self.resources = resources or []
        self.prompts = prompts or []

    def to_dict(self):
        return {
            "name": self.name,
            "transport": self.transport,
            "tools": self.tools,
            "resources": self.resources,
            "prompts": self.prompts
        }
