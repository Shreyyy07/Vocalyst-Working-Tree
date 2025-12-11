/**
 * Suppresses Three.js deprecation warnings that come from React Three Fiber
 * This is needed because R3F uses deprecated APIs internally
 */
export function suppressThreeJsWarnings() {
    if (typeof window !== 'undefined') {
        // Store the original console.error
        const originalError = console.error;

        // Override console.error and console.warn to filter out specific Three.js warnings
        const overrideConsole = (originalFn: any) => {
            return (...args: any[]) => {
                const message = args[0]?.toString() || '';
                // Filter out the outputEncoding deprecation warning
                if (message.includes('Property .outputEncoding has been removed')) {
                    return;
                }
                originalFn.apply(console, args);
            };
        };

        console.error = overrideConsole(originalError);
        console.warn = overrideConsole(console.warn);
    }
}
