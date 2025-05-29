import { readFileSync, writeFileSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import sharp from "sharp";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Get command line arguments
const args = process.argv.slice(2);
const revision = args[0] || "0.0.1";

console.log(`🎨 Compiling SVG banner with revision: ${revision}`);

// Read the SVG template
const svgPath = join(__dirname, "../public/xarray-banner.svg");
let svgContent;

try {
    svgContent = readFileSync(svgPath, "utf8");
    console.log("✅ SVG template loaded successfully");
} catch (error) {
    console.error("❌ Error reading SVG file:", error.message);
    process.exit(1);
}

// Replace the revision placeholder with the actual revision
const updatedSvg = svgContent.replace(/REV: [^<]+/, `REV: ${revision}`);

// Create the inline SVG with proper dimensions and transparent background
const inlineSvg = `
<svg width="1280" height="640" viewBox="0 0 1280 640" xmlns="http://www.w3.org/2000/svg">
  <!-- Transparent background -->
  <rect width="1280" height="640" fill="transparent" rx="19.2" ry="19.2" />

  <!-- Dashed border with rounded corners -->
  <rect
    x="16"
    y="16"
    width="1248"
    height="608"
    fill="none"
    stroke="#4A90E2"
    strokeWidth="3.2"
    strokeDasharray="12.8,6.4"
    opacity="0.7"
    rx="16"
    ry="16"
  />

  <!-- Blueprint grid pattern -->
  <defs>
    <pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse">
      <path d="M 32 0 L 0 0 0 32" fill="none" stroke="#4A90E2" strokeWidth="0.8" opacity="0.3" />
    </pattern>

    <!-- Hatching patterns for different logo components -->
    <pattern id="hatch-dark-blue" patternUnits="userSpaceOnUse" width="12.8" height="12.8">
      <path d="M0,12.8 L12.8,0" stroke="#216C89" strokeWidth="1.28" opacity="0.3" />
    </pattern>

    <pattern id="hatch-medium-blue" patternUnits="userSpaceOnUse" width="9.6" height="9.6">
      <path d="M0,9.6 L9.6,0" stroke="#4993AA" strokeWidth="0.96" opacity="0.25" />
    </pattern>

    <pattern id="hatch-light-blue" patternUnits="userSpaceOnUse" width="16" height="16">
      <path d="M0,16 L16,0" stroke="#0F4565" strokeWidth="1.6" opacity="0.2" />
    </pattern>

    <pattern id="hatch-cyan" patternUnits="userSpaceOnUse" width="11.2" height="11.2">
      <path d="M0,0 L11.2,11.2" stroke="#6BE8E8" strokeWidth="1.12" opacity="0.3" />
    </pattern>

    <pattern id="hatch-light-cyan" patternUnits="userSpaceOnUse" width="8" height="8">
      <path d="M0,0 L8,8" stroke="#9DEEF4" strokeWidth="0.8" opacity="0.25" />
    </pattern>

    <pattern id="hatch-teal" patternUnits="userSpaceOnUse" width="14.4" height="14.4">
      <path d="M0,0 L14.4,14.4" stroke="#4ACFDD" strokeWidth="1.28" opacity="0.2" />
    </pattern>

    <pattern id="hatch-orange" patternUnits="userSpaceOnUse" width="9.6" height="9.6">
      <path d="M0,0 L9.6,9.6 M0,9.6 L9.6,0" stroke="#E38017" strokeWidth="0.96" opacity="0.3" />
    </pattern>

    <pattern id="hatch-green" patternUnits="userSpaceOnUse" width="12.8" height="12.8">
      <path d="M0,0 L12.8,12.8" stroke="#16AFB5" strokeWidth="1.12" opacity="0.25" />
    </pattern>
  </defs>
  <rect width="1280" height="640" fill="url(#grid)" rx="19.2" ry="19.2" />

  <!-- Corner markers with rounded caps -->
  <g stroke="#4A90E2" strokeWidth="3.2" fill="none" opacity="0.8" strokeLinecap="round">
    <!-- Top left -->
    <path d="M 40 40 L 72 40 M 40 40 L 40 72" />
    <!-- Top right -->
    <path d="M 1240 40 L 1208 40 M 1240 40 L 1240 72" />
    <!-- Bottom left -->
    <path d="M 40 600 L 72 600 M 40 600 L 40 568" />
    <!-- Bottom right -->
    <path d="M 1240 600 L 1208 600 M 1240 600 L 1240 568" />
  </g>

  <!-- GitHub URL in upper left corner -->
  <text
    x="80"
    y="56"
    fontFamily="'Geist Mono', 'Courier New', monospace"
    fontSize="16"
    fill="#4A90E2"
    textAnchor="start"
  >
    srivarra/xrdantic
  </text>

  <!-- Version/revision info in bottom right corner -->
  <text
    x="1200"
    y="584"
    fontFamily="'Geist Mono', 'Courier New', monospace"
    fontSize="16"
    fill="#4A90E2"
    textAnchor="end"
  >
    REV: ${revision}
  </text>

  <!-- Xarray logo positioned and centered on the left -->
  <g transform="translate(120, 120) scale(0.65)">
    <g>
      <!-- Bottom section - dark blue -->
      <polygon
        points="266.62,546.18 356.1,454.54 356.1,271.27 266.62,362.9"
        fill="url(#hatch-dark-blue)"
        stroke="#216C89"
        strokeWidth="6.4"
        strokeDasharray="12.8,6.4"
        strokeLinejoin="round"
      />
      <polygon
        points="356.1,271.45 114.48,271.45 25,362.9 266.62,362.9"
        fill="url(#hatch-medium-blue)"
        stroke="#4993AA"
        strokeWidth="6.4"
        strokeDasharray="12.8,6.4"
        strokeLinejoin="round"
      />
      <rect
        x="25"
        y="362.9"
        width="241.62"
        height="183.27"
        fill="url(#hatch-light-blue)"
        stroke="#0F4565"
        strokeWidth="6.4"
        strokeDasharray="12.8,6.4"
        rx="12.8"
        ry="12.8"
      />
    </g>
    <g>
      <!-- Top section - cyan -->
      <polygon
        points="266.62,328.73 356.1,237.1 356.1,53.82 266.62,145.46"
        fill="url(#hatch-cyan)"
        stroke="#6BE8E8"
        strokeWidth="6.4"
        strokeDasharray="12.8,6.4"
        strokeLinejoin="round"
      />
      <polygon
        points="356.1,54 114.48,54 25,145.46 266.62,145.46"
        fill="url(#hatch-light-cyan)"
        stroke="#9DEEF4"
        strokeWidth="6.4"
        strokeDasharray="12.8,6.4"
        strokeLinejoin="round"
      />
      <rect
        x="25"
        y="145.46"
        width="241.62"
        height="183.27"
        fill="url(#hatch-teal)"
        stroke="#4ACFDD"
        strokeWidth="6.4"
        strokeDasharray="12.8,6.4"
        rx="12.8"
        ry="12.8"
      />
    </g>
    <!-- Right side elements -->
    <polygon
      points="467.47,452.33 374.48,546.18 374.48,362.9 467.47,269.05"
      fill="url(#hatch-orange)"
      stroke="#E38017"
      strokeWidth="6.4"
      strokeDasharray="12.8,6.4"
      strokeLinejoin="round"
    />
    <polygon
      points="575,452.33 482.01,546.18 482.01,362.9 575,269.05"
      fill="url(#hatch-green)"
      stroke="#16AFB5"
      strokeWidth="6.4"
      strokeDasharray="12.8,6.4"
      strokeLinejoin="round"
    />
  </g>

  <!-- Centered blueprint-style title and subtitle -->
  <g transform="translate(640, 160)">
    <text
      x="0"
      y="0"
      fontFamily="'Geist Mono', 'Courier New', monospace"
      fontSize="64"
      fontWeight="bold"
      fill="#2C5282"
      textAnchor="middle"
    >
      XRDANTIC
    </text>
    <text
      x="0"
      y="40"
      fontFamily="'Geist Mono', 'Courier New', monospace"
      fontSize="28"
      fill="#4A90E2"
      textAnchor="middle"
    >
      Pydantic Scaffolding for Xarray
    </text>
  </g>

  <!-- Centered 2x2 Grid Layout for Data Type Icons -->
  <g transform="translate(640, 280)">
    <!-- Grid background centered -->
    <rect
      x="-280"
      y="-20"
      width="560"
      height="280"
      fill="none"
      stroke="#4A90E2"
      strokeWidth="1.6"
      strokeDasharray="4.8,4.8"
      opacity="0.3"
      rx="6.4"
    />

    <!-- Top Left: Coordinates -->
    <g transform="translate(-140, 50)">
      <g transform="translate(0, -18)">
        <line x1="-60" y1="0" x2="60" y2="0" stroke="#6BE8E8" strokeWidth="4" strokeDasharray="12,6" />
        <circle cx="-60" cy="0" r="6" fill="#6BE8E8" stroke="#6BE8E8" strokeWidth="2" />
        <circle cx="-30" cy="0" r="4" fill="none" stroke="#6BE8E8" strokeWidth="2" />
        <circle cx="0" cy="0" r="4" fill="none" stroke="#6BE8E8" strokeWidth="2" />
        <circle cx="30" cy="0" r="4" fill="none" stroke="#6BE8E8" strokeWidth="2" />
        <circle cx="60" cy="0" r="6" fill="#6BE8E8" stroke="#6BE8E8" strokeWidth="2" />
        <text
          x="0"
          y="35"
          fontFamily="'Geist Mono', 'Courier New', monospace"
          fontSize="16"
          fill="#4A90E2"
          textAnchor="middle"
        >
          Coordinates
        </text>
      </g>
    </g>

    <!-- Top Right: DataArray -->
    <g transform="translate(140, 50)">
      <g transform="translate(0, -24)">
        <rect x="-45" y="-12" width="24" height="24" fill="url(#hatch-medium-blue)" stroke="#4993AA" strokeWidth="2" strokeDasharray="6,3" rx="3" />
        <rect x="-12" y="-12" width="24" height="24" fill="url(#hatch-medium-blue)" stroke="#4993AA" strokeWidth="2" strokeDasharray="6,3" rx="3" />
        <rect x="21" y="-12" width="24" height="24" fill="url(#hatch-medium-blue)" stroke="#4993AA" strokeWidth="2" strokeDasharray="6,3" rx="3" />
        <rect x="-45" y="18" width="24" height="24" fill="url(#hatch-medium-blue)" stroke="#4993AA" strokeWidth="2" strokeDasharray="6,3" rx="3" />
        <rect x="-12" y="18" width="24" height="24" fill="url(#hatch-medium-blue)" stroke="#4993AA" strokeWidth="2" strokeDasharray="6,3" rx="3" />
        <rect x="21" y="18" width="24" height="24" fill="url(#hatch-medium-blue)" stroke="#4993AA" strokeWidth="2" strokeDasharray="6,3" rx="3" />
        <text
          x="0"
          y="65"
          fontFamily="'Geist Mono', 'Courier New', monospace"
          fontSize="16"
          fill="#4A90E2"
          textAnchor="middle"
        >
          DataArrays
        </text>
      </g>
    </g>

    <!-- Bottom Left: Dataset -->
    <g transform="translate(-140, 170)">
      <g transform="translate(0, -30)">
        <!-- Back layer cubes -->
        <rect x="-24" y="-12" width="20" height="20" fill="url(#hatch-light-blue)" stroke="#0F4565" strokeWidth="2" strokeDasharray="6,3" rx="3" opacity="0.7" />
        <rect x="4" y="-12" width="20" height="20" fill="url(#hatch-light-blue)" stroke="#0F4565" strokeWidth="2" strokeDasharray="6,3" rx="3" opacity="0.7" />
        <rect x="-24" y="16" width="20" height="20" fill="url(#hatch-light-blue)" stroke="#0F4565" strokeWidth="2" strokeDasharray="6,3" rx="3" opacity="0.7" />
        <rect x="4" y="16" width="20" height="20" fill="url(#hatch-light-blue)" stroke="#0F4565" strokeWidth="2" strokeDasharray="6,3" rx="3" opacity="0.7" />
        <!-- Front layer cubes -->
        <rect x="-36" y="0" width="20" height="20" fill="url(#hatch-dark-blue)" stroke="#216C89" strokeWidth="2" strokeDasharray="6,3" rx="3" />
        <rect x="-8" y="0" width="20" height="20" fill="url(#hatch-dark-blue)" stroke="#216C89" strokeWidth="2" strokeDasharray="6,3" rx="3" />
        <rect x="-36" y="28" width="20" height="20" fill="url(#hatch-dark-blue)" stroke="#216C89" strokeWidth="2" strokeDasharray="6,3" rx="3" />
        <rect x="-8" y="28" width="20" height="20" fill="url(#hatch-dark-blue)" stroke="#216C89" strokeWidth="2" strokeDasharray="6,3" rx="3" />
        <!-- 3D connection lines -->
        <line x1="-16" y1="0" x2="-4" y2="-12" stroke="#4A90E2" strokeWidth="1.5" opacity="0.6" />
        <line x1="12" y1="0" x2="24" y2="-12" stroke="#4A90E2" strokeWidth="1.5" opacity="0.6" />
        <line x1="-16" y1="28" x2="-4" y2="16" stroke="#4A90E2" strokeWidth="1.5" opacity="0.6" />
        <line x1="12" y1="28" x2="24" y2="16" stroke="#4A90E2" strokeWidth="1.5" opacity="0.6" />
        <text
          x="0"
          y="75"
          fontFamily="'Geist Mono', 'Courier New', monospace"
          fontSize="16"
          fill="#4A90E2"
          textAnchor="middle"
        >
          Datasets
        </text>
      </g>
    </g>

    <!-- Bottom Right: DataTree -->
    <g transform="translate(140, 170)">
      <g transform="translate(0, -30)">
        <!-- Root node -->
        <rect x="-16" y="-8" width="32" height="16" fill="url(#hatch-orange)" stroke="#E38017" strokeWidth="2" strokeDasharray="6,3" rx="3" />
        <!-- Branch lines -->
        <line x1="0" y1="8" x2="0" y2="24" stroke="#16AFB5" strokeWidth="3" strokeDasharray="8,4" />
        <line x1="-32" y1="24" x2="32" y2="24" stroke="#16AFB5" strokeWidth="3" strokeDasharray="8,4" />
        <line x1="-32" y1="24" x2="-32" y2="40" stroke="#16AFB5" strokeWidth="3" strokeDasharray="8,4" />
        <line x1="0" y1="24" x2="0" y2="40" stroke="#16AFB5" strokeWidth="3" strokeDasharray="8,4" />
        <line x1="32" y1="24" x2="32" y2="40" stroke="#16AFB5" strokeWidth="3" strokeDasharray="8,4" />
        <!-- Child nodes -->
        <rect x="-44" y="40" width="24" height="16" fill="url(#hatch-green)" stroke="#16AFB5" strokeWidth="2" strokeDasharray="6,3" rx="3" />
        <rect x="-12" y="40" width="24" height="16" fill="url(#hatch-green)" stroke="#16AFB5" strokeWidth="2" strokeDasharray="6,3" rx="3" />
        <rect x="20" y="40" width="24" height="16" fill="url(#hatch-green)" stroke="#16AFB5" strokeWidth="2" strokeDasharray="6,3" rx="3" />
        <text
          x="0"
          y="85"
          fontFamily="'Geist Mono', 'Courier New', monospace"
          fontSize="16"
          fill="#4A90E2"
          textAnchor="middle"
        >
          DataTrees
        </text>
      </g>
    </g>
  </g>
</svg>
`;

console.log("🔄 Converting SVG to PNG with transparency...");

try {
    // Convert SVG to PNG using Sharp with transparency
    const pngBuffer = await sharp(Buffer.from(inlineSvg))
        .png({
            compressionLevel: 9,
            adaptiveFiltering: true,
            force: true,
        })
        .toBuffer();

    // Write the PNG file
    const outputPath = join(__dirname, "../dist/xrdantic-banner.png");
    writeFileSync(outputPath, pngBuffer);

    console.log(`✅ PNG banner generated successfully!`);
    console.log(`📁 Output: ${outputPath}`);
    console.log(`📏 Dimensions: 1280x640px`);
    console.log(`🎯 Revision: ${revision}`);
    console.log(`💾 File size: ${(pngBuffer.length / 1024).toFixed(2)} KB`);
} catch (error) {
    console.error("❌ Error converting SVG to PNG:", error.message);
    process.exit(1);
}
