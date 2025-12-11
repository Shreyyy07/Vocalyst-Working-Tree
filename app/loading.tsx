export default function Loading() {
    return (
        <div className="min-h-screen flex items-center justify-center bg-black">
            <div className="text-center space-y-6">
                <div className="w-16 h-16 mx-auto border-4 border-cyan-500 border-t-transparent rounded-full animate-spin" />
                <p className="text-xl text-white/80 font-light">
                    Loading...
                </p>
            </div>
        </div>
    );
}
