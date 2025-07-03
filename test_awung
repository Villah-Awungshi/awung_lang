#!/usr/bin/env python3
"""
AwungLang - A Bible-inspired AI-first programming language
Complete implementation with lexer, parser, interpreter, and AI integration
"""

import re
import json
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum


class TokenType(Enum):
    BIBLE_CODE = "BIBLE_CODE"
    IDENTIFIER = "IDENTIFIER"
    STRING = "STRING"
    NUMBER = "NUMBER"
    SYMBOL = "SYMBOL"
    OPERATOR = "OPERATOR"
    NEWLINE = "NEWLINE"
    EOF = "EOF"
    LBRACE = "LBRACE"
    RBRACE = "RBRACE"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    LBRACKET = "LBRACKET"
    RBRACKET = "RBRACKET"


@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int


class AwungLexer:
    """Lexer for AwungLang - converts source code into tokens"""
    
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens = []
        
        # Define patterns
        self.patterns = {
            'BIBLE_CODE': r'\d{1,3}:\d{1,3}',
            'STRING': r'"[^"]*"',
            'NUMBER': r'#\d+(?:\.\d+)?',
            'SYMBOL': r"'[^']*'",
            'IDENTIFIER': r'[a-zA-Z_][a-zA-Z0-9_]*',
            'OPERATOR': r'(\+|IS|EQUALS|GREATER|LESS)',
            'WHITESPACE': r'[ \t]+',
            'NEWLINE': r'\n',
            'COMMENT': r'//.*',
        }
        
        # Compile patterns
        self.regex = {}
        for name, pattern in self.patterns.items():
            self.regex[name] = re.compile(pattern)
    
    def current_char(self) -> Optional[str]:
        if self.pos >= len(self.source):
            return None
        return self.source[self.pos]
    
    def peek_char(self, offset: int = 1) -> Optional[str]:
        peek_pos = self.pos + offset
        if peek_pos >= len(self.source):
            return None
        return self.source[peek_pos]
    
    def advance(self):
        if self.pos < len(self.source) and self.source[self.pos] == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        self.pos += 1
    
    def match_pattern(self, pattern_name: str) -> Optional[str]:
        regex = self.regex[pattern_name]
        match = regex.match(self.source, self.pos)
        if match:
            return match.group(0)
        return None
    
    def tokenize(self) -> List[Token]:
        while self.pos < len(self.source):
            # Skip whitespace
            if self.match_pattern('WHITESPACE'):
                whitespace = self.match_pattern('WHITESPACE')
                self.pos += len(whitespace)
                self.column += len(whitespace)
                continue
            
            # Skip comments
            if self.match_pattern('COMMENT'):
                comment = self.match_pattern('COMMENT')
                self.pos += len(comment)
                continue
            
            # Match tokens in order of priority
            matched = False
            
            for pattern_name in ['BIBLE_CODE', 'STRING', 'NUMBER', 'SYMBOL', 'OPERATOR', 'IDENTIFIER', 'NEWLINE']:
                match = self.match_pattern(pattern_name)
                if match:
                    token_type = TokenType(pattern_name)
                    token = Token(token_type, match, self.line, self.column)
                    self.tokens.append(token)
                    
                    for _ in range(len(match)):
                        self.advance()
                    matched = True
                    break
            
            # Handle special characters
            if not matched:
                char = self.current_char()
                if char in '{}()[]':
                    token_map = {
                        '{': TokenType.LBRACE,
                        '}': TokenType.RBRACE,
                        '(': TokenType.LPAREN,
                        ')': TokenType.RPAREN,
                        '[': TokenType.LBRACKET,
                        ']': TokenType.RBRACKET,
                    }
                    token = Token(token_map[char], char, self.line, self.column)
                    self.tokens.append(token)
                    self.advance()
                else:
                    # Skip unknown characters
                    self.advance()
        
        # Add EOF token
        self.tokens.append(Token(TokenType.EOF, '', self.line, self.column))
        return self.tokens


class ASTNode:
    """Base class for AST nodes"""
    pass


@dataclass
class DeclarationNode(ASTNode):
    bible_code: str
    identifier: str
    value: Any


@dataclass
class FunctionCallNode(ASTNode):
    bible_code: str
    function_name: str
    target: str
    arguments: List[Any]


@dataclass
class ExpressionNode(ASTNode):
    operator: str
    left: Any
    right: Any


@dataclass
class ConditionalNode(ASTNode):
    condition: Any
    then_block: List[ASTNode]
    else_block: Optional[List[ASTNode]] = None


@dataclass
class LoopNode(ASTNode):
    condition: Any
    body: List[ASTNode]


class AwungParser:
    """Parser for AwungLang - converts tokens into AST"""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[0] if tokens else None
        
        # Bible code to command mapping
        self.bible_commands = {
            '1:1': 'DECLARE',
            '19:14': 'PROCLAIM',
            '40:31': 'DIVINE_VISION',
            '15:5': 'SPEAK',
            '16:9': 'DRAW',
            '23:1': 'PROTECT',
            '22:21': 'RETURN',
            '7:12': 'END',
            '10:1': 'BEGIN',
            '11:11': 'RECEIVE',
            '3:16': 'JUDGE',      # if statement
            '7:7': 'REPEAT',      # loop
            '12:12': 'REFUGE',    # try/catch
        }
    
    def advance(self):
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = Token(TokenType.EOF, '', 0, 0)
    
    def peek(self, offset: int = 1) -> Optional[Token]:
        peek_pos = self.pos + offset
        if peek_pos < len(self.tokens):
            return self.tokens[peek_pos]
        return None
    
    def parse_expression(self) -> Any:
        """Parse expressions with operators"""
        left = self.parse_primary()
        
        while (self.current_token.type == TokenType.OPERATOR and 
               self.current_token.value in ['+', 'EQUALS', 'GREATER', 'LESS']):
            operator = self.current_token.value
            self.advance()
            right = self.parse_primary()
            left = ExpressionNode(operator, left, right)
        
        return left
    
    def parse_primary(self) -> Any:
        """Parse primary expressions"""
        if self.current_token.type == TokenType.STRING:
            value = self.current_token.value[1:-1]  # Remove quotes
            self.advance()
            return value
        elif self.current_token.type == TokenType.NUMBER:
            value = float(self.current_token.value[1:])  # Remove # prefix
            self.advance()
            return value
        elif self.current_token.type == TokenType.SYMBOL:
            value = self.current_token.value[1:-1]  # Remove quotes
            self.advance()
            return value
        elif self.current_token.type == TokenType.IDENTIFIER:
            identifier = self.current_token.value
            self.advance()
            return identifier
        else:
            raise SyntaxError(f"Unexpected token: {self.current_token.value}")
    
    def parse_statement(self) -> Optional[ASTNode]:
        """Parse a single statement"""
        if self.current_token.type == TokenType.BIBLE_CODE:
            bible_code = self.current_token.value
            command = self.bible_commands.get(bible_code, 'UNKNOWN')
            self.advance()
            
            if command == 'DECLARE':
                # 1:1 identifier value
                if self.current_token.type != TokenType.IDENTIFIER:
                    raise SyntaxError(f"Expected identifier after {bible_code}")
                identifier = self.current_token.value
                self.advance()
                
                value = self.parse_expression()
                return DeclarationNode(bible_code, identifier, value)
            
            elif command in ['PROCLAIM', 'SPEAK', 'DIVINE_VISION', 'DRAW', 'PROTECT', 'RECEIVE']:
                # Function calls
                if self.current_token.type != TokenType.IDENTIFIER:
                    raise SyntaxError(f"Expected identifier after {bible_code}")
                target = self.current_token.value
                self.advance()
                
                # Handle function arguments
                arguments = []
                if self.current_token.type == TokenType.LPAREN:
                    self.advance()  # Skip (
                    while self.current_token.type != TokenType.RPAREN:
                        arguments.append(self.parse_expression())
                        if self.current_token.type == TokenType.OPERATOR and self.current_token.value == ',':
                            self.advance()
                    self.advance()  # Skip )
                else:
                    # Single argument or expression
                    if self.current_token.type not in [TokenType.NEWLINE, TokenType.EOF]:
                        arguments.append(self.parse_expression())
                
                return FunctionCallNode(bible_code, command, target, arguments)
            
            elif command == 'JUDGE':
                # Conditional statement
                condition = self.parse_expression()
                then_block = []
                else_block = None
                
                # Parse then block (simplified)
                while (self.current_token.type not in [TokenType.EOF, TokenType.NEWLINE] and
                       not (self.current_token.type == TokenType.BIBLE_CODE and 
                            self.current_token.value == '7:12')):
                    stmt = self.parse_statement()
                    if stmt:
                        then_block.append(stmt)
                
                return ConditionalNode(condition, then_block, else_block)
            
            elif command == 'REPEAT':
                # Loop statement
                condition = self.parse_expression()
                body = []
                
                # Parse body (simplified)
                while (self.current_token.type not in [TokenType.EOF, TokenType.NEWLINE] and
                       not (self.current_token.type == TokenType.BIBLE_CODE and 
                            self.current_token.value == '7:12')):
                    stmt = self.parse_statement()
                    if stmt:
                        body.append(stmt)
                
                return LoopNode(condition, body)
        
        return None
    
    def parse(self) -> List[ASTNode]:
        """Parse the entire program"""
        ast = []
        
        while self.current_token.type != TokenType.EOF:
            # Skip newlines
            if self.current_token.type == TokenType.NEWLINE:
                self.advance()
                continue
            
            stmt = self.parse_statement()
            if stmt:
                ast.append(stmt)
            else:
                self.advance()  # Skip unrecognized tokens
        
        return ast


class AIEngine:
    """Mock AI engine for AwungLang AI functions"""
    
    def __init__(self):
        self.model_name = "GPT-10-Divine"
    
    def divine_vision(self, prompt: str) -> str:
        """Generate text based on prompt"""
        return f"[AI Vision] Generated content for: '{prompt}' - A mystical tale unfolds..."
    
    def speak(self, text: str) -> str:
        """Convert text to speech"""
        return f"[AI Voice] Speaking: '{text}'"
    
    def draw(self, prompt: str) -> str:
        """Generate image/art from prompt"""
        return f"[AI Art] Generated artwork for: '{prompt}' - A beautiful digital masterpiece"
    
    def protect(self, input_text: str) -> str:
        """Scan for cybersecurity threats"""
        threats = ["virus", "malware", "phishing", "hack", "exploit"]
        detected = [t for t in threats if t.lower() in input_text.lower()]
        if detected:
            return f"[Security Alert] Threats detected: {', '.join(detected)}"
        return "[Security] No threats detected - System is protected"
    
    def receive_input(self, prompt: str = "") -> str:
        """Simulate user input"""
        return f"[User Input] Received: '{prompt or 'Hello, AwungLang!'}'"


class AwungInterpreter:
    """Interpreter for AwungLang - executes the AST"""
    
    def __init__(self):
        self.variables = {}
        self.ai_engine = AIEngine()
        self.output_buffer = []
    
    def evaluate_expression(self, expr: Any) -> Any:
        """Evaluate expressions"""
        if isinstance(expr, str):
            # Check if it's a variable
            return self.variables.get(expr, expr)
        elif isinstance(expr, (int, float)):
            return expr
        elif isinstance(expr, ExpressionNode):
            left = self.evaluate_expression(expr.left)
            right = self.evaluate_expression(expr.right)
            
            if expr.operator == '+':
                return str(left) + str(right)
            elif expr.operator == 'EQUALS':
                return left == right
            elif expr.operator == 'GREATER':
                return left > right
            elif expr.operator == 'LESS':
                return left < right
        
        return expr
    
    def execute_node(self, node: ASTNode):
        """Execute a single AST node"""
        if isinstance(node, DeclarationNode):
            # Variable declaration
            value = self.evaluate_expression(node.value)
            self.variables[node.identifier] = value
            self.output_buffer.append(f"[{node.bible_code}] Declared {node.identifier} = {value}")
        
        elif isinstance(node, FunctionCallNode):
            # Function call
            if node.function_name == 'PROCLAIM':
                # Output
                value = self.variables.get(node.target, node.target)
                self.output_buffer.append(f"[PROCLAIM] {value}")
            
            elif node.function_name == 'DIVINE_VISION':
                # AI text generation
                if node.arguments:
                    prompt = self.evaluate_expression(node.arguments[0])
                else:
                    prompt = self.variables.get(node.target, node.target)
                
                result = self.ai_engine.divine_vision(str(prompt))
                self.variables[node.target] = result
                self.output_buffer.append(result)
            
            elif node.function_name == 'SPEAK':
                # AI voice
                value = self.variables.get(node.target, node.target)
                result = self.ai_engine.speak(str(value))
                self.output_buffer.append(result)
            
            elif node.function_name == 'DRAW':
                # AI image generation
                if node.arguments:
                    prompt = self.evaluate_expression(node.arguments[0])
                else:
                    prompt = self.variables.get(node.target, node.target)
                
                result = self.ai_engine.draw(str(prompt))
                self.variables[node.target] = result
                self.output_buffer.append(result)
            
            elif node.function_name == 'PROTECT':
                # AI security scan
                if node.arguments:
                    target = self.evaluate_expression(node.arguments[0])
                else:
                    target = self.variables.get(node.target, node.target)
                
                result = self.ai_engine.protect(str(target))
                self.variables[node.target] = result
                self.output_buffer.append(result)
            
            elif node.function_name == 'RECEIVE':
                # User input
                prompt = node.arguments[0] if node.arguments else ""
                result = self.ai_engine.receive_input(str(prompt))
                self.variables[node.target] = result
                self.output_buffer.append(result)
        
        elif isinstance(node, ConditionalNode):
            # Conditional execution
            condition = self.evaluate_expression(node.condition)
            if condition:
                for stmt in node.then_block:
                    self.execute_node(stmt)
            elif node.else_block:
                for stmt in node.else_block:
                    self.execute_node(stmt)
        
        elif isinstance(node, LoopNode):
            # Loop execution
            while self.evaluate_expression(node.condition):
                for stmt in node.body:
                    self.execute_node(stmt)
    
    def interpret(self, ast: List[ASTNode]) -> List[str]:
        """Interpret the entire AST"""
        self.output_buffer = []
        
        for node in ast:
            try:
                self.execute_node(node)
            except Exception as e:
                self.output_buffer.append(f"[ERROR] {str(e)}")
        
        return self.output_buffer


class AwungLang:
    """Main AwungLang interpreter class"""
    
    def __init__(self):
        self.lexer = None
        self.parser = None
        self.interpreter = AwungInterpreter()
    
    def run(self, source_code: str) -> List[str]:
        """Execute AwungLang source code"""
        try:
            # Tokenize
            self.lexer = AwungLexer(source_code)
            tokens = self.lexer.tokenize()
            
            # Parse
            self.parser = AwungParser(tokens)
            ast = self.parser.parse()
            
            # Interpret
            output = self.interpreter.interpret(ast)
            
            return output
            
        except Exception as e:
            return [f"[RUNTIME ERROR] {str(e)}"]
    
    def debug_tokens(self, source_code: str) -> List[Token]:
        """Debug: show tokens"""
        self.lexer = AwungLexer(source_code)
        return self.lexer.tokenize()
    
    def debug_ast(self, source_code: str) -> List[ASTNode]:
        """Debug: show AST"""
        self.lexer = AwungLexer(source_code)
        tokens = self.lexer.tokenize()
        self.parser = AwungParser(tokens)
        return self.parser.parse()


# Example usage and test
if __name__ == "__main__":
    # Test program
    awung_code = '''
    // The Last Dream of Earth - A AwungLang Story
    1:1 title "The Last Dream of Earth"
    1:1 genre "Sci-Fi"
    1:1 author "AwungLang AI"
    
    // Generate movie summary using AI
    40:31 summary "Write a movie summary for " + title + " in " + genre + " style"
    
    // Output the results
    19:14 title
    19:14 genre
    19:14 summary
    
    // Speak the summary
    15:5 summary
    
    // Generate concept art
    16:9 artwork "Scene concept for: " + summary
    
    // Security scan
    23:1 security_check "Scan this AI application for threats"
    
    // Get user input
    11:11 user_feedback "What did you think of the story?"
    19:14 user_feedback
    '''
    
    # Create interpreter and run
    awung = AwungLang()
    
    print("=== AwungLang Interpreter ===")
    print("Running sample program...\n")
    
    # Execute the program
    output = awung.run(awung_code)
    
    # Display output
    for line in output:
        print(line)
    
    print("\n=== Execution Complete ===")
    
    # Debug information
    print("\n=== Debug Information ===")
    print("Variables:", awung.interpreter.variables)