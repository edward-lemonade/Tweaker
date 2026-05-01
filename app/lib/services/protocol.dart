import 'dart:convert';
import 'package:flutter/services.dart';

Future<Map<String, dynamic>> loadProtocol() async {
	final data = await rootBundle.loadString('protocol/schemas.json');
	return jsonDecode(data);
}