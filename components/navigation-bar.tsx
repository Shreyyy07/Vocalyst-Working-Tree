// "use client";

// import Link from "next/link";
// import { usePathname } from "next/navigation";
// import { cn } from "@/lib/utils";

// const navItems = [
//   { href: "/", label: "Home" },
//   { href: "/practice", label: "Practice" },
//   { href: "/camera", label: "Camera" },
//   { href: "/tts", label: "TTS Lab" },
//   { href: "/analytics", label: "Analytics" },
// ];

// export function NavigationBar() {
//   const pathname = usePathname();

//   return (
//     <nav className="border-b">
//       <div className="container flex h-16 items-center px-4">
//         <Link href="/" className="mr-6 flex items-center space-x-2">
//           <span className="text-xl font-bold">Vocalyst</span>
//         </Link>
//         <div className="flex gap-6">
//           {navItems.map((item) => (
//             <Link
//               key={item.href}
//               href={item.href}
//               className={cn(
//                 "transition-colors hover:text-foreground/80",
//                 pathname === item.href
//                   ? "text-foreground"
//                   : "text-foreground/60"
//               )}
//             >
//               {item.label}
//             </Link>
//           ))}
//         </div>
//       </div>
//     </nav>
//   );
// }
"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const navItems = [
  { href: "/", label: "Home" },
  { href: "/practice", label: "Practice" },
  { href: "/camera", label: "Camera" },
  { href: "/tts", label: "TTS Lab" },
  { href: "/speech-to-text", label: "STT Lab" },
  { href: "/analytics", label: "Analytics" },
];

export function NavigationBar() {
  const pathname = usePathname();

  return (
    <nav className="fixed top-0 left-0 w-full z-50 bg-black/40 backdrop-blur-xl border-b border-white/10 shadow-lg">
      <div className="container max-w-7xl mx-auto flex h-20 items-center justify-between px-6">
        {/* Brand */}
        <Link href="/" className="flex items-center gap-2 group">
          <span className="text-3xl font-extrabold bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 text-transparent bg-clip-text tracking-tight transition-all duration-300 group-hover:bg-gradient-to-l">
            Vocalyst
          </span>
        </Link>

        {/* Navigation Links */}
        <div className="flex items-center gap-8">
          {navItems.map((item) => {
            const isActive = pathname === item.href;

            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "relative text-base font-medium px-4 py-2 rounded-full transition-all duration-300",
                  isActive
                    ? "text-cyan-400 bg-cyan-400/10"
                    : "text-white/70 hover:text-white hover:bg-white/5"
                )}
              >
                <span>{item.label}</span>
                {isActive && (
                  <span className="absolute bottom-0 left-1/2 w-1.5 h-1.5 bg-cyan-400 rounded-full transform -translate-x-1/2 -mb-2 shadow-[0_0_10px_rgba(34,211,238,0.8)]" />
                )}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}