"use client";

import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useMemo, useRef, useEffect } from "react";
import { suppressThreeJsWarnings } from "@/lib/suppressThreeWarnings";

function FloatingPoints() {
  const particlesRef = useRef<THREE.Points>(null);

  // Create particles
  const particles = useMemo(() => {
    const particleCount = 5000;
    const positions = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount; i++) {
      const i3 = i * 3;
      positions[i3] = THREE.MathUtils.randFloatSpread(2000);     // x
      positions[i3 + 1] = THREE.MathUtils.randFloatSpread(2000); // y
      positions[i3 + 2] = THREE.MathUtils.randFloatSpread(2000); // z
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    // Simple circular texture for particles
    const textureSize = 64;
    const data = new Uint8Array(textureSize * textureSize * 4);

    // Create a simple circular gradient
    for (let i = 0; i < textureSize; i++) {
      for (let j = 0; j < textureSize; j++) {
        const index = (i * textureSize + j) * 4;

        // Calculate distance from center (normalized)
        const x = i / textureSize - 0.5;
        const y = j / textureSize - 0.5;
        const distance = Math.sqrt(x * x + y * y) * 2; // *2 to normalize to 0-1 range

        // Value based on distance (1 at center, 0 at edge)
        let value = Math.max(0, 1 - distance);
        value = value * value; // Square for smoother falloff

        data[index] = 255;     // R
        data[index + 1] = 255; // G
        data[index + 2] = 255; // B
        data[index + 3] = value * 255; // A
      }
    }

    const texture = new THREE.DataTexture(
      data,
      textureSize,
      textureSize,
      THREE.RGBAFormat
    );
    texture.needsUpdate = true;

    const material = new THREE.PointsMaterial({
      size: 6,
      map: texture,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      sizeAttenuation: true,
      color: 0xffffff
    });

    return new THREE.Points(geometry, material);
  }, []);

  useFrame(() => {
    if (particlesRef.current) {
      particlesRef.current.rotation.y += 0.0005;
      particlesRef.current.rotation.x += 0.0003;
    }
  });

  return <primitive ref={particlesRef} object={particles} />;
}

export default function FloatingPointsCanvas() {
  // Suppress Three.js deprecation warnings from React Three Fiber
  useEffect(() => {
    suppressThreeJsWarnings();
  }, []);

  return (
    <Canvas
      className="fixed inset-0 z-0"
      camera={{ position: [0, 0, 1000], fov: 75 }}
      gl={{
        antialias: true,
        alpha: true,
      }}
    >
      <FloatingPoints />
    </Canvas>
  );
}