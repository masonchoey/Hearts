/**
 * WebSocket-based multiplayer game state hook.
 * Connects to the backend WS endpoint and keeps game state in sync.
 */
import { useState, useEffect, useRef, useCallback } from 'react'
import { createGameWebSocket } from '../api/multiplayer'

export function useMultiplayerGame(roomId, token) {
  const [gameState, setGameState] = useState(null)
  const [mySeat, setMySeat] = useState(null)
  const [connectionStatus, setConnectionStatus] = useState('connecting') // connecting | open | closed | error
  const [error, setError] = useState(null)
  const [players, setPlayers] = useState([]) // lobby players list (before game starts)
  const [connectedSeats, setConnectedSeats] = useState([])

  const wsRef = useRef(null)
  const reconnectTimeout = useRef(null)

  const connect = useCallback(() => {
    if (!roomId || !token) return
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    setConnectionStatus('connecting')
    setConnectedSeats([])
    const ws = createGameWebSocket(roomId, token)
    wsRef.current = ws

    ws.onopen = () => {
      setConnectionStatus('open')
      setError(null)
    }

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data)

        switch (msg.type) {
          case 'game_started':
            setGameState(msg.state)
            setMySeat(msg.state.my_seat)
            break

          case 'game_state':
            setGameState(msg.state)
            if (msg.state.my_seat !== undefined) setMySeat(msg.state.my_seat)
            break

          case 'player_joined':
            setPlayers(prev => {
              const existing = prev.find(p => p.seat === msg.player.seat)
              if (existing) return prev
              return [...prev, msg.player]
            })
            break

          case 'connection_status':
            setConnectedSeats(msg.connected_seats ?? [])
            break

          case 'player_disconnected':
            if (msg.connected_seats) {
              setConnectedSeats(msg.connected_seats)
            }
            break

          case 'error':
            setError(msg.message)
            break

          case 'pong':
            break

          default:
            break
        }
      } catch (e) {
        console.warn('WS parse error', e)
      }
    }

    ws.onerror = () => {
      setConnectionStatus('error')
      setError('Connection error')
    }

    ws.onclose = (evt) => {
      setConnectionStatus('closed')
      // Reconnect after 3 s unless we chose to close
      if (evt.code !== 1000 && evt.code !== 4001 && evt.code !== 4003) {
        reconnectTimeout.current = setTimeout(connect, 3000)
      }
    }
  }, [roomId, token])

  useEffect(() => {
    connect()
    return () => {
      clearTimeout(reconnectTimeout.current)
      wsRef.current?.close(1000)
    }
  }, [connect])

  const playCard = useCallback((card) => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) {
      setError('Not connected')
      return
    }
    wsRef.current.send(JSON.stringify({ type: 'play_card', card }))
  }, [])

  const passCards = useCallback((cards) => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) {
      setError('Not connected')
      return
    }
    wsRef.current.send(JSON.stringify({ type: 'pass_cards', cards }))
  }, [])

  const ping = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'ping' }))
    }
  }, [])

  const send = useCallback((payload) => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) {
      setError('Not connected')
      return
    }
    wsRef.current.send(JSON.stringify(payload))
  }, [])

  const nextRound = useCallback(() => send({ type: 'next_round' }), [send])
  const subInAI = useCallback((seat) => send({ type: 'sub_ai', seat }), [send])
  const endMatch = useCallback(() => send({ type: 'end_match' }), [send])

  return {
    gameState,
    mySeat,
    connectionStatus,
    error,
    players,
    connectedSeats,
    playCard,
    passCards,
    ping,
    nextRound,
    subInAI,
    endMatch,
  }
}
