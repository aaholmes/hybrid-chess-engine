//! Tests for InferenceServer mock constructors and predict_async

use kingfisher::board::Board;
use kingfisher::mcts::InferenceServer;

#[test]
fn test_mock_server_returns_result() {
    let server = InferenceServer::new_mock();
    let board = Board::new();
    let receiver = server.predict_async(board);
    let result = receiver.recv().expect("Should receive a response");
    assert!(result.is_some(), "Mock server should return Some");
    let (policy, value, k) = result.unwrap();
    assert_eq!(policy.len(), 4672, "Policy should have 4672 elements");
    assert!(value >= -1.0 && value <= 1.0, "Value {value} should be in [-1, 1]");
    assert!(k >= 0.0 && k <= 1.0, "k={k} should be in [0, 1]");
}

#[test]
fn test_mock_biased_returns_correct_value() {
    let server = InferenceServer::new_mock_biased(0.5);
    let board = Board::new();
    let receiver = server.predict_async(board);
    let result = receiver.recv().expect("Should receive a response");
    let (_, value, k) = result.unwrap();
    assert!(
        (value - 0.5).abs() < 1e-6,
        "Biased mock should return value=0.5, got {value}"
    );
    assert!(
        (k - 0.5).abs() < 1e-6,
        "Biased mock should return k=0.5, got {k}"
    );
}

#[test]
fn test_mock_server_handles_multiple_requests() {
    let server = InferenceServer::new_mock();
    let board = Board::new();

    for i in 0..5 {
        let receiver = server.predict_async(board.clone());
        let result = receiver.recv().expect(&format!("Request {i} should get response"));
        assert!(result.is_some(), "Request {i} should return Some");
    }
}

#[test]
fn test_mock_server_policy_length() {
    let server = InferenceServer::new_mock_biased(0.0);
    let board = Board::new();
    let receiver = server.predict_async(board);
    let (policy, _, _) = receiver.recv().unwrap().unwrap();
    assert_eq!(
        policy.len(),
        4672,
        "Policy vector should have exactly 4672 elements"
    );
}
