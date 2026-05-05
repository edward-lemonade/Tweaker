enum HandPose { flat, pinch, point, idle, away }

class Hand {
	final double x, y;
  final double vx, vy;
  final double theta;
	final HandPose pose;

	const Hand({
		required this.x,
		required this.y,
		required this.vx,
		required this.vy,
    required this.theta,
		required this.pose,
	});

	factory Hand.fromJson(Map<String, dynamic> j) => Hand(
		x: (j['x'] as num).toDouble(),
		y: (j['y'] as num).toDouble(),
		vx: (j['vx'] as num).toDouble(),
		vy: (j['vy'] as num).toDouble(),
    theta: (j['theta'] as num).toDouble(),
		pose: HandPose.values.byName((j['pose'] as String).toLowerCase()),
	);

	Map<String, dynamic> toJson() => {
		'x': x, 
    'y': y, 
    'vx': vx, 
    'vy': vy,
    'theta': theta,
		'pose': pose.name.toUpperCase(),
	};
}