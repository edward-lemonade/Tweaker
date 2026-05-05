import 'dart:async';
import 'dart:convert';
import 'dart:io';

import '../models/hand.dart';
import '../models/events.dart';

class HandService {
	HandService({
		this.port = 5555,
		required this.pythonDir,
	});

	final int port;
	final String pythonDir;

	Process? _process;
	RawDatagramSocket? _socket;

	final _controller = StreamController<GestureEvent>.broadcast();

	Stream<GestureEvent> get stream => _controller.stream;
	bool get isRunning => _process != null;

	// Start the Python service and begin receiving gesture events.
	Future<void> start() async {
		if (_process != null) return;

		_process = await Process.start(
			'.venv/bin/python',
			['main.py', '--port', '$port'],
			workingDirectory: pythonDir,
		);

		_process!.stdout.transform(utf8.decoder).listen(
			(line) => print('[python stdout] $line'),
		);
		_process!.stderr.transform(utf8.decoder).listen(
			(line) => print('[python stderr] $line'),
		);

		_process!.exitCode.then((code) {
			print('[hand_service] Python exited: $code');
			_process = null;
		});

		// Give Python a moment to open the camera before binding
		await Future.delayed(const Duration(milliseconds: 500));

		_socket = await RawDatagramSocket.bind(InternetAddress.loopbackIPv4, port);
		_socket!.listen((event) {
			if (event != RawSocketEvent.read) return;
			final dg = _socket!.receive();
			if (dg == null) return;
			try {
				final json = jsonDecode(utf8.decode(dg.data)) as Map<String, dynamic>;
				_onPacket(json);
			} catch (_) {}
		});
	}

	// Python sends: { "type": "GESTURE", "leftHand": {…}, "rightHand": {…}, "timestamp": … }
	void _onPacket(Map<String, dynamic> json) {
		if (json['type'] != 'GESTURE') return;

		_controller.add(GestureEvent(
			leftHand:  Hand.fromJson(json['leftHand']  as Map<String, dynamic>),
			rightHand: Hand.fromJson(json['rightHand'] as Map<String, dynamic>),
			timestamp: (json['timestamp'] as num).toDouble(),
		));
	}

	// Stop receiving and kill the Python process.
	Future<void> stop() async {
		_socket?.close();
		_socket = null;
		_process?.kill();
		_process = null;
		await Future.delayed(const Duration(milliseconds: 100));
	}

	void dispose() {
		stop();
		_controller.close();
	}
}