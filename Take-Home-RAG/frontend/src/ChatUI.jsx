import React, { useState } from 'react';
import './ChatUI.css';

const ChatUI = () => {
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');

  const parseRawInput = (rawInput) => {
    const lines = rawInput.split('\n').filter(line => line.trim() !== '');

    const questionLine = lines.find(line => line.toLowerCase().startsWith('q:')) || lines[0];
    const question = questionLine.replace(/^q[:.]?\s*/i, '').trim();

    const optionsLines = lines.filter(line => /^[A-D][.)]/i.test(line.trim()));
    const options = optionsLines.join('\n').trim();

    return { question, options };
  };

  const handleSend = async (e) => {
    e.preventDefault();
    if (!userInput.trim()) return;

    const { question, options } = parseRawInput(userInput);

    const newMessages = [
      ...messages,
      { type: 'user', text: userInput }
    ];
    setMessages(newMessages);
    setUserInput('');

    const response = await fetch('http://localhost:8000/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, options }),
    });

    const data = await response.json();
    setMessages([
      ...newMessages,
      { type: 'bot', text: data.answer }
    ]);
  };

  return (
    <div className="chat-container">
      <div className="chat-messages">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`message ${msg.type === 'user' ? 'user-message' : 'bot-message'}`}
          >
            {msg.text}
          </div>
        ))}
      </div>

      <form className="chat-input" onSubmit={handleSend}>
        <textarea
          placeholder="Type your input here..."
          rows={4}
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
        />
        <button type="submit">Send</button>
      </form>
    </div>
  );
};

export default ChatUI;
