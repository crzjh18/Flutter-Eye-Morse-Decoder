import 'dart:async';
import 'dart:typed_data';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:tflite_flutter/tflite_flutter.dart'; 
import 'package:flutter_tts/flutter_tts.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:audioplayers/audioplayers.dart'; 

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MaterialApp(
    debugShowCheckedModeBanner: false,
    home: RobustBlinkPage()
  ));
}

class RobustBlinkPage extends StatefulWidget {
  const RobustBlinkPage({super.key});
  @override
  State<RobustBlinkPage> createState() => _RobustBlinkPageState();
}

class _RobustBlinkPageState extends State<RobustBlinkPage> {
  // --- CONTROLLERS ---
  CameraController? _controller;
  final FaceDetector _faceDetector = FaceDetector(
    options: FaceDetectorOptions(
      enableLandmarks: true,
      enableClassification: true, 
      enableTracking: false,
    ),
  );
  
  final FlutterTts _tts = FlutterTts();
  final AudioPlayer _audioPlayer = AudioPlayer(); 
  Interpreter? _cnnInterpreter; 
  
  // --- STATE ---
  bool _isProcessing = false;
  String _eyeState = "OPEN";
  String _decodedText = "";
  String _currentSequence = ""; 
  String _wordBuffer = ""; 
  
  // --- SETTINGS ---
  bool _speakCharacters = true; // 1. Char Voice
  bool _speakWords = true;      // 2. Word Voice
  bool _soundEnabled = true;    // 3. Beep Sound
  bool _debugMode = false;      // 4. "Nerd Mode" (Show raw scores)
  
  // --- DEBUG DATA (For the Panel) ---
  double _debugClosedScore = 0.0;
  double _debugOpenScore = 0.0;
  
  // --- LOGIC VARIABLES ---
  bool _isClosed = false;
  int _closedStart = 0;
  int _lastBlinkEndMs = 0;
  bool _spaceInserted = false; 
  
  // --- TUNING ---
  final int _dashThresholdMs = 400;   
  final int _deleteThresholdMs = 2000; 
  final int _letterGapMs = 1500;      // 1.5s = Letter
  final int _wordGapMs = 5000;        // 5.0s = Space
  
  // SMOOTHING BUFFER
  final List<bool> _blinkHistory = []; 
  final int _historySize = 5;          
  
  final Float32List _inputBuffer = Float32List(64 * 64);

  // --- MORSE MAP ---
  final Map<String, String> _morseMap = {
    // Letters
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E",
    "..-.": "F", "--.": "G", "....": "H", "..": "I", ".---": "J",
    "-.-": "K", ".-..": "L", "--": "M", "-.": "N", "---": "O",
    ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T",
    "..-": "U", "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y",
    "--..": "Z",
    // Numbers
    ".----": "1", "..---": "2", "...--": "3", "....-": "4", ".....": "5",
    "-....": "6", "--...": "7", "---..": "8", "----.": "9", "-----": "0",
    // Symbols
    ".-.-.-": ".", "--..--": ",", 
  };

  // --- USER SHORTCUTS (RESTORED) ---
  final Map<String, String> _shortcuts = {
    "----": "HELP",
    ".....": "WATER",
  };

  @override
  void initState() {
    super.initState();
    _startSystem();
  }

  Future<void> _startSystem() async {
    await Permission.camera.request();
    
    // PRELOAD AUDIO
    try {
       await _audioPlayer.setSource(AssetSource('beep.mp3'));
    } catch(e) {
       print("⚠️ Audio Warning: assets/beep.mp3 missing.");
    }
    
    try {
      _cnnInterpreter = await Interpreter.fromAsset('assets/eye_state_cnn.tflite');
      print("✅ Thesis CNN Model Loaded");
    } catch (e) {
      print("❌ Model Load Error: $e");
    }

    final cameras = await availableCameras();
    final frontCam = cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.front,
      orElse: () => cameras.first
    );

    _controller = CameraController(
      frontCam,
      ResolutionPreset.low, 
      enableAudio: false,
      imageFormatGroup: Platform.isAndroid ? ImageFormatGroup.nv21 : ImageFormatGroup.bgra8888,
    );

    await _controller!.initialize();
    await _tts.setLanguage("en-US");
    await _tts.setPitch(1.0);
    
    Timer.periodic(const Duration(milliseconds: 200), (timer) {
       _checkSilence();
    });

    if (mounted) setState(() {});
    _controller!.startImageStream(_processFrame);
  }

  void _processFrame(CameraImage image) async {
    if (_isProcessing) return;
    _isProcessing = true;

    try {
      final inputImage = _convertInputImage(image);
      if (inputImage == null) return;
      
      final faces = await _faceDetector.processImage(inputImage);
      if (faces.isEmpty) { _isProcessing = false; return; }

      final face = faces.first;
      
      // --- 1. FOCUS LOCK (Prevents Ghosting) ---
      double rotY = face.headEulerAngleY ?? 0.0;
      if (rotY.abs() > 30) {
        _updateBlinkLogic(false); 
        _isProcessing = false; 
        return; 
      }
      
      // --- 2. CNN INFERENCE (The Thesis Model) ---
      bool cnnSaysClosed = false;
      final leftEyePos = face.landmarks[FaceLandmarkType.leftEye];
      
      if (leftEyePos != null && _cnnInterpreter != null) {
         _cropEyePixels(image, leftEyePos.position.x.toInt(), leftEyePos.position.y.toInt());
         var input = _inputBuffer.reshape([1, 64, 64, 1]);
         var output = List.filled(2, 0.0).reshape([1, 2]);
         _cnnInterpreter!.run(input, output);
         
         double scoreClosed = output[0][0];
         double scoreOpen = output[0][1];

         // DEBUG: Update UI Variables
         if (_debugMode && mounted) {
            setState(() {
              _debugClosedScore = scoreClosed;
              _debugOpenScore = scoreOpen;
            });
         }
         
         // Decision Logic
         if (scoreClosed > scoreOpen) {
           cnnSaysClosed = true;
         }
      }

      // --- 3. ML KIT (Safety Net) ---
      bool mlKitSaysClosed = false;
      if (face.leftEyeOpenProbability != null) {
        if (face.leftEyeOpenProbability! < 0.4) {
          mlKitSaysClosed = true;
        }
      }

      // --- 4. ENSEMBLE MERGE ---
      bool finalDecision = cnnSaysClosed || mlKitSaysClosed;
      
      _blinkHistory.add(finalDecision);
      if (_blinkHistory.length > 5) _blinkHistory.removeAt(0);
      int closedVotes = _blinkHistory.where((c) => c).length;
      bool smoothed = closedVotes > 2; 

      _updateBlinkLogic(smoothed);

    } catch (e) {
      print("Error: $e");
    } finally {
      _isProcessing = false;
    }
  }

  void _updateBlinkLogic(bool isClosed) {
    int now = DateTime.now().millisecondsSinceEpoch;
    
    if (isClosed) {
      _spaceInserted = false; 
      if (!_isClosed) {
        _isClosed = true;
        _closedStart = now;
        if (mounted) setState(() => _eyeState = "CLOSED");
      } else {
        int duration = now - _closedStart;
        if (duration > _deleteThresholdMs && _eyeState != "DELETING") {
           setState(() => _eyeState = "DELETING");
        }
      }
    } else {
      if (_isClosed) {
        int duration = now - _closedStart;
        _isClosed = false;
        _lastBlinkEndMs = now; 
        
        if (mounted) setState(() => _eyeState = "OPEN");
        
        if (duration > _deleteThresholdMs) {
           _triggerBackspace(); 
        } else if (duration > 100) { 
           String signal = (duration > _dashThresholdMs) ? "-" : ".";
           _playSignalSound(signal);
           setState(() {
             _currentSequence += signal;
           });
        }
      }
    }
  }

  // --- SOUND LOGIC (Robust) ---
  void _playSignalSound(String signal) async {
    if (!_soundEnabled) return; 

    try {
      await _audioPlayer.stop(); 
      await _audioPlayer.play(AssetSource('beep.mp3'));
      
      // Short beep for Dot (150ms), Long for Dash (400ms)
      int waitTime = (signal == ".") ? 150 : 400; 
      
      await Future.delayed(Duration(milliseconds: waitTime));
      await _audioPlayer.stop();
    } catch(e) {
      print("Audio Error: $e");
    }
  }

  void _triggerBackspace() {
    print("ACTION: Backspace");
    if (_speakCharacters || _speakWords) _tts.speak("Delete"); 
    
    setState(() {
       _currentSequence = ""; 
       if (_decodedText.isNotEmpty) {
         _decodedText = _decodedText.substring(0, _decodedText.length - 1);
         if (_wordBuffer.isNotEmpty) {
            _wordBuffer = _wordBuffer.substring(0, _wordBuffer.length - 1);
         }
       }
    });
  }

  void _checkSilence() {
    // FIX: If eyes are closed, do NOT count silence (Prevents false Space)
    if (_isClosed) return; 

    int now = DateTime.now().millisecondsSinceEpoch;
    int silenceDuration = now - _lastBlinkEndMs;

    // 1. LETTER GAP
    if (_currentSequence.isNotEmpty && silenceDuration > _letterGapMs) {
       _translateAndSpeak();
    }

    // 2. WORD GAP (Auto Space)
    if (_decodedText.isNotEmpty && 
        !_decodedText.endsWith(" ") && 
        !_spaceInserted && 
        silenceDuration > _wordGapMs) {
       _triggerAutoSpace();
    }
  }

  void _triggerAutoSpace() {
    setState(() {
      _decodedText += " ";
      _spaceInserted = true; 
      
      // FIX: Speak Lowercase so it sounds like a word
      if (_speakWords && _wordBuffer.isNotEmpty) {
         _tts.speak(_wordBuffer.toLowerCase());
      }
      _wordBuffer = "";
    });
  }

  void _translateAndSpeak() {
    // 1. CHECK FOR USER SHORTCUTS FIRST (RESTORED)
    if (_shortcuts.containsKey(_currentSequence)) {
      String phrase = _shortcuts[_currentSequence]!;
      
      setState(() {
        _decodedText += "$phrase "; // Add space automatically after a shortcut
        _currentSequence = "";
        _spaceInserted = true; // Prevent double spacing
      });
      
      // Always speak the full phrase immediately
      _tts.speak(phrase);
      return; // Exit function, don't do normal letter logic
    }

    // 2. NORMAL MORSE TRANSLATION
    String letter = _morseMap[_currentSequence] ?? "?";
    
    setState(() {
      _decodedText += letter;
      if (letter != "?") {
        _wordBuffer += letter; 
      }
      _currentSequence = "";
      _spaceInserted = false; 
    });
    
    if (letter != "?" && _speakCharacters) {
      _tts.speak(letter);
    }
  }
  
  // --- SETTINGS PANEL ---
  Widget _buildSettingsPanel() {
    return Column(
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            _buildToggle("Char Voice", _speakCharacters, (val) => setState(() => _speakCharacters = val)),
            _buildToggle("Word Voice", _speakWords, (val) => setState(() => _speakWords = val)),
            _buildToggle("Beep Sound", _soundEnabled, (val) => setState(() => _soundEnabled = val)),
          ],
        ),
        
        const SizedBox(height: 10),
        
        // --- NEW BUTTON LOCATION: CENTERED & LARGE ---
        SizedBox(
          width: double.infinity,
          child: ElevatedButton.icon(
            icon: const Icon(Icons.edit_note, color: Colors.white),
            label: const Text("MANAGE SHORTCUTS", style: TextStyle(fontWeight: FontWeight.bold)),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.green, // Bright green
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(vertical: 12),
            ),
            onPressed: () => _showShortcutManager(context),
          ),
        ),

        // DEBUG TOGGLE (Hidden feature for Defense)
        TextButton(
          onPressed: () => setState(() => _debugMode = !_debugMode),
          child: Text(
            _debugMode ? "HIDE MODEL DATA" : "SHOW MODEL DATA",
            style: const TextStyle(fontSize: 10, color: Colors.grey),
          ),
        )
      ],
    );
  }

  Widget _buildToggle(String label, bool value, Function(bool) onChanged) {
    return Column(
      children: [
        Text(label, style: const TextStyle(color: Colors.white, fontSize: 12)),
        Switch(
          value: value,
          activeColor: Colors.green,
          inactiveThumbColor: Colors.red,
          inactiveTrackColor: Colors.red[200],
          onChanged: onChanged,
        ),
      ],
    );
  }

  // --- NEW: MANAGE SHORTCUTS POPUP (RESTORED) ---
  void _showShortcutManager(BuildContext context) {
    TextEditingController seqController = TextEditingController();
    TextEditingController wordController = TextEditingController();

    showModalBottomSheet(
      context: context,
      isScrollControlled: true, 
      backgroundColor: Colors.grey[900],
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) {
        return StatefulBuilder( 
          builder: (BuildContext context, StateSetter setModalState) {
            return Padding(
              padding: EdgeInsets.only(
                bottom: MediaQuery.of(context).viewInsets.bottom,
                left: 20, right: 20, top: 20
              ),
              child: Container(
                height: 600,
                child: Column(
                  children: [
                    const Text("Custom Shortcuts", 
                      style: TextStyle(color: Colors.white, fontSize: 22, fontWeight: FontWeight.bold)
                    ),
                    const SizedBox(height: 15),
                    
                    // INPUT FIELDS
                    Row(
                      children: [
                        Expanded(
                          flex: 1,
                          child: TextField(
                            controller: seqController,
                            style: const TextStyle(color: Colors.white),
                            decoration: const InputDecoration(
                              labelText: "Code (....)",
                              labelStyle: TextStyle(color: Colors.grey),
                              enabledBorder: OutlineInputBorder(borderSide: BorderSide(color: Colors.grey)),
                            ),
                          ),
                        ),
                        const SizedBox(width: 10),
                        Expanded(
                          flex: 2,
                          child: TextField(
                            controller: wordController,
                            style: const TextStyle(color: Colors.white),
                            decoration: const InputDecoration(
                              labelText: "Word (WATER)",
                              labelStyle: TextStyle(color: Colors.grey),
                              enabledBorder: OutlineInputBorder(borderSide: BorderSide(color: Colors.grey)),
                            ),
                          ),
                        ),
                        IconButton(
                          icon: const Icon(Icons.add_circle, color: Colors.green, size: 30),
                          onPressed: () {
                            if (seqController.text.isNotEmpty && wordController.text.isNotEmpty) {
                              setState(() {
                                _shortcuts[seqController.text] = wordController.text;
                              });
                              setModalState(() {
                                seqController.clear();
                                wordController.clear();
                              });
                            }
                          },
                        )
                      ],
                    ),
                    const SizedBox(height: 20),
                    
                    // LIST OF EXISTING SHORTCUTS
                    Expanded(
                      child: ListView.builder(
                        itemCount: _shortcuts.length,
                        itemBuilder: (context, index) {
                          String key = _shortcuts.keys.elementAt(index);
                          String value = _shortcuts.values.elementAt(index);
                          return Card(
                            color: Colors.grey[800],
                            margin: const EdgeInsets.symmetric(vertical: 5),
                            child: ListTile(
                              title: Text(value, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
                              subtitle: Text("Code: $key", style: const TextStyle(color: Colors.cyanAccent)),
                              trailing: IconButton(
                                icon: const Icon(Icons.delete, color: Colors.red),
                                onPressed: () {
                                  setState(() {
                                    _shortcuts.remove(key);
                                  });
                                  setModalState(() {}); 
                                },
                              ),
                            ),
                          );
                        },
                      ),
                    ),
                  ],
                ),
              ),
            );
          }
        );
      },
    );
  }

  // --- NEW: CHEATSHEET POPUP ---
  void _showCheatsheet(BuildContext context) {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.grey[900],
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) {
        return Container(
          padding: const EdgeInsets.all(20),
          height: 500, // Half screen height
          child: Column(
            children: [
              const Text("Morse Code Guide", 
                style: TextStyle(color: Colors.white, fontSize: 24, fontWeight: FontWeight.bold)
              ),
              const SizedBox(height: 10),
              Expanded(
                child: GridView.builder(
                  gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                    crossAxisCount: 3, // 3 Columns
                    childAspectRatio: 2.5, // Wide rectangles
                    crossAxisSpacing: 10,
                    mainAxisSpacing: 10,
                  ),
                  itemCount: _morseMap.length,
                  itemBuilder: (context, index) {
                    String key = _morseMap.keys.elementAt(index);
                    String value = _morseMap.values.elementAt(index);
                    return Container(
                      decoration: BoxDecoration(
                        color: Colors.grey[800],
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(color: Colors.cyan.withOpacity(0.3))
                      ),
                      child: Center(
                        child: Text("$value  $key", 
                          style: const TextStyle(color: Colors.cyanAccent, fontSize: 16, fontWeight: FontWeight.bold)
                        ),
                      ),
                    );
                  },
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  // --- IMAGE HELPERS ---
  void _cropEyePixels(CameraImage image, int centerX, int centerY) {
    final int width = image.width;
    final int height = image.height;
    final Uint8List yPlane = image.planes[0].bytes; 
    
    int startX = centerX - 32;
    int startY = centerY - 32;
    int bufferIndex = 0;

    for (int y = 0; y < 64; y++) {
      for (int x = 0; x < 64; x++) {
        int px = startX + x;
        int py = startY + y;
        if (px < 0) px = 0; if (px >= width) px = width - 1;
        if (py < 0) py = 0; if (py >= height) py = height - 1;
        int pixel = yPlane[py * width + px];
        _inputBuffer[bufferIndex++] = pixel / 255.0;
      }
    }
  }

  InputImage? _convertInputImage(CameraImage image) {
    final sensorOrient = _controller!.description.sensorOrientation;
    InputImageRotation rotation = InputImageRotation.rotation0deg;
    if (sensorOrient == 90) rotation = InputImageRotation.rotation90deg;
    else if (sensorOrient == 270) rotation = InputImageRotation.rotation270deg;
    else if (sensorOrient == 180) rotation = InputImageRotation.rotation180deg;

    return InputImage.fromBytes(
      bytes: _concatenatePlanes(image.planes),
      metadata: InputImageMetadata(
        size: Size(image.width.toDouble(), image.height.toDouble()),
        rotation: rotation,
        format: InputImageFormat.nv21,
        bytesPerRow: image.planes[0].bytesPerRow,
      ),
    );
  }

  Uint8List _concatenatePlanes(List<Plane> planes) {
    final allBytes = WriteBuffer();
    for (final plane in planes) {
      allBytes.putUint8List(plane.bytes);
    }
    return allBytes.done().buffer.asUint8List();
  }

  @override
  void dispose() {
    _controller?.dispose();
    _faceDetector.close();
    _cnnInterpreter?.close();
    _audioPlayer.dispose(); 
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      
      appBar: AppBar(
        title: const Text("Thesis Hybrid System"), 
        backgroundColor: Colors.grey[900],
        actions: [
          IconButton(
            icon: const Icon(Icons.help_outline, color: Colors.cyanAccent),
            onPressed: () => _showCheatsheet(context),
            tooltip: "Show Cheatsheet",
          )
        ],
      ),

      body: Column(
        children: [
          // 1. CAMERA PREVIEW (Reduced Height)
          Expanded(
            flex: 1, // WAS 3, NOW 1 (Takes 50% of screen)
            child: Stack(
              children: [
                _controller != null && _controller!.value.isInitialized
                  ? Container(width: double.infinity, child: CameraPreview(_controller!))
                  : const Center(child: CircularProgressIndicator()),
                
                if (_debugMode)
                  Positioned(
                    bottom: 10, left: 10, right: 10,
                    child: Container(
                      padding: const EdgeInsets.all(8),
                      color: Colors.black.withOpacity(0.7),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text("🧠 CNN RAW LOGITS", style: TextStyle(color: Colors.cyan, fontSize: 10, fontWeight: FontWeight.bold)),
                          Text("CLOSED: ${_debugClosedScore.toStringAsFixed(2)}", style: const TextStyle(color: Colors.red, fontSize: 12)),
                          LinearProgressIndicator(value: (_debugClosedScore + 10) / 20, color: Colors.red),
                          const SizedBox(height: 4),
                          Text("OPEN:   ${_debugOpenScore.toStringAsFixed(2)}", style: const TextStyle(color: Colors.green, fontSize: 12)),
                          LinearProgressIndicator(value: (_debugOpenScore + 10) / 20, color: Colors.green),
                        ],
                      ),
                    ),
                  )
              ],
            ),
          ),
          
          // 2. CONTROL PANEL (Increased Height)
          Expanded(
            flex: 1, // WAS 2, NOW 1 (Takes 50% of screen)
            child: Container(
              width: double.infinity,
              color: Colors.grey[850],
              padding: const EdgeInsets.all(10), 
              child: Column(
                mainAxisAlignment: MainAxisAlignment.spaceBetween, // Distribute space evenly
                children: [
                  // Top Section: Settings & Status
                  Column(
                    children: [
                      _buildSettingsPanel(),
                      const Divider(color: Colors.grey),
                      Text(_eyeState, style: TextStyle(
                        fontSize: 28, fontWeight: FontWeight.bold,
                        color: _eyeState == "CLOSED" ? Colors.red : 
                               (_eyeState == "DELETING" ? Colors.orange : Colors.green)
                      )),
                      Container(
                        margin: const EdgeInsets.only(top: 5),
                        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 5),
                        decoration: BoxDecoration(color: Colors.blue[900], borderRadius: BorderRadius.circular(10)),
                        child: Text("SEQ: $_currentSequence", style: const TextStyle(fontSize: 20, color: Colors.white)),
                      ),
                    ],
                  ),
                  
                  // Middle Section: The Text Output (Flexible)
                  Expanded(
                    child: Center(
                      child: SingleChildScrollView( // Allows scrolling if text gets too long
                        child: Text(
                          _decodedText.isEmpty ? "..." : _decodedText,
                          textAlign: TextAlign.center,
                          style: const TextStyle(fontSize: 32, color: Colors.yellowAccent, fontWeight: FontWeight.bold),
                        ),
                      ),
                    ),
                  ),
                  
                  // Bottom Section: The Buttons (Fixed at bottom of panel)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 10), // Add padding from bottom edge
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      children: [
                        ElevatedButton.icon(
                          onPressed: () {
                             _tts.speak(_decodedText.isEmpty ? "No text" : _decodedText);
                          },
                          icon: const Icon(Icons.volume_up),
                          label: const Text("SPEAK"),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.green,
                            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15)
                          ),
                        ),
                        ElevatedButton.icon(
                          onPressed: () => setState(() { _decodedText = ""; _currentSequence = ""; _wordBuffer = ""; _spaceInserted = false; }),
                          icon: const Icon(Icons.delete_forever),
                          label: const Text("RESET"),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.red,
                            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15)
                          ),
                        ),
                      ],
                    ),
                  )
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}