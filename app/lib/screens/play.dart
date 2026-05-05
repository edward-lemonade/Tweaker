import 'dart:async';

import 'package:flutter/material.dart';

import '../models/events.dart';
import '../models/hand.dart';
import '../services/hand_service.dart';

// =====================================================================
// Screen

class PlayScreen extends StatefulWidget {
	const PlayScreen({super.key, required this.handService});

	final HandService handService;

	@override
	State<PlayScreen> createState() => _PlayScreenState();
}

// =====================================================================
// State

class _PlayScreenState extends State<PlayScreen> {
	GestureEvent? _latest;
	StreamSubscription<GestureEvent>? _sub;

	@override
	void initState() {
		super.initState();
		widget.handService.start();
		_sub = widget.handService.stream.listen((event) {
			if (mounted) setState(() => _latest = event);
		});
	}

	@override
	void dispose() {
		_sub?.cancel();
		widget.handService.stop();
		super.dispose();
	}

	@override
	Widget build(BuildContext context) {
		return Scaffold(
			backgroundColor: Colors.black,
			body: LayoutBuilder(
				builder: (context, constraints) {
					final size = constraints.biggest;
					return Stack(
						children: [
							if (_latest != null) ...[
								if (_latest!.leftHand.pose != HandPose.away)
									_HandWidget(hand: _latest!.leftHand, size: size),
								if (_latest!.rightHand.pose != HandPose.away)
									_HandWidget(hand: _latest!.rightHand, size: size),
							],
						],
					);
				},
			),
		);
	}
}

// =====================================================================
// Hand Widget

class _HandWidget extends StatelessWidget {
	const _HandWidget({required this.hand, required this.size});

	final Hand hand;
	final Size size;

	Offset get _offset => Offset(hand.x * size.width, hand.y * size.height);

	@override
  Widget build(BuildContext context) {
    const diameter = 64.0;
    return AnimatedPositioned(
      duration: const Duration(milliseconds: 32),
      curve: Curves.easeOut,
      left: _offset.dx - diameter / 2,
      top:  _offset.dy - diameter / 2,
      child: Transform.rotate(
        angle: hand.theta,
        child: Container(
          width: diameter,
          height: diameter,
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.9),
            shape: BoxShape.circle,
          ),
          child: const Icon(Icons.back_hand, size: 32),
        ),
      ),
    );
  }
}