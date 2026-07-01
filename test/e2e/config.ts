export const TESTAPP_URL = 'https://localhost:4200/';
export const RUN_MODE = process.env['RUN_MODE'] || 'development';

// LiveKit connection details for the OpenVidu local deployment used in tests
export const LIVEKIT_URL_HTTP = 'http://localhost:7880';
export const LIVEKIT_URL_RTC = 'ws://localhost:7880';
export const LIVEKIT_API_KEY = 'devkey';
export const LIVEKIT_API_SECRET = 'secret';