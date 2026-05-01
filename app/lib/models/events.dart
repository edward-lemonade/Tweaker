import 'hand.dart';

class GestureEvent {
	final Hand leftHand;
	final Hand rightHand;
	final double timestamp;

	const GestureEvent({
		required this.leftHand,
		required this.rightHand,
		required this.timestamp,
	});

	factory GestureEvent.fromJson(Map<String, dynamic> j) => GestureEvent(
		leftHand: Hand.fromJson(j['leftHand']),
		rightHand: Hand.fromJson(j['rightHand']),
		timestamp: (j['timestamp'] as num).toDouble(),
	);
}

class AudioControlEvent {
	final String action;
	final int deck;
	final double value;
	final double timestamp;

	const AudioControlEvent({
		required this.action,
		required this.deck,
		required this.value,
		required this.timestamp,
	});

	Map<String, dynamic> toJson() => {
		'action': action,
		'deck': deck,
		'value': value,
		'timestamp': timestamp,
	};
}

class AudioFeedbackEvent {
	final String type;
	final int deck;
	final Map<String, dynamic> data;
	final double timestamp;

	const AudioFeedbackEvent({
		required this.type,
		required this.deck,
		required this.data,
		required this.timestamp,
	});

	factory AudioFeedbackEvent.fromJson(Map<String, dynamic> j) => AudioFeedbackEvent(
		type: j['type'],
		deck: j['deck'],
		data: j['data'],
		timestamp: (j['timestamp'] as num).toDouble(),
	);
}

class SystemEvent {
	final String type;
	final double timestamp;

	const SystemEvent({required this.type, required this.timestamp});

	factory SystemEvent.fromJson(Map<String, dynamic> j) => SystemEvent(
		type: j['type'],
		timestamp: (j['timestamp'] as num).toDouble(),
	);

	Map<String, dynamic> toJson() => {'type': type, 'timestamp': timestamp};
}