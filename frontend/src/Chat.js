import React, { useState } from 'react';
import axios from 'axios';

function Chat() {
    const [input, setInput] = useState("");
    const [response, setResponse] = useState("");

    const sendMessage = async () => {
        try {
            const res = await axios.post("http://localhost:8000/chat", { question: input });
            setResponse(res.data.response);
        } catch (error) {
            setResponse("Error: Unable to get a response.");
        }
    };

    return (
        <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
            <h1>RAG Chatbot</h1>
            <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask something..."
                style={{ padding: '10px', width: '300px', marginRight: '10px' }}
            />
            <button
                onClick={sendMessage}
                style={{ padding: '10px', backgroundColor: '#007BFF', color: 'white', border: 'none', cursor: 'pointer' }}
            >
                Send
            </button>
            <div style={{ marginTop: '20px' }}>
                <strong>Response:</strong>
                <p>{response}</p>
            </div>
        </div>
    );
}

export default Chat;