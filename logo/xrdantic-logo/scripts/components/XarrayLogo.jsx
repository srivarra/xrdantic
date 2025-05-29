import React from "react";

const XarrayLogo = ({ colors }) => {
    // Ensure colors are available, falling back to defaults if necessary
    const c = {
        darkBlue: "#216C89",
        mediumBlue: "#4993AA",
        lightBlue: "#0F4565",
        cyan: "#6BE8E8",
        lightCyan: "#9DEEF4",
        teal: "#4ACFDD",
        orange: "#E38017",
        green: "#16AFB5",
        ...colors, // Override defaults with passed colors
    };

    return (
        <g transform="translate(120, 120) scale(0.65)">
            {/* Bottom section - dark blue */}
            <polygon
                points="266.62,546.18 356.1,454.54 356.1,271.27 266.62,362.9"
                fill="url(#hatch-dark-blue)"
                stroke={c.darkBlue}
                strokeWidth="6.4"
                strokeDasharray="12.8,6.4"
                strokeLinejoin="round"
            />
            <polygon
                points="356.1,271.45 114.48,271.45 25,362.9 266.62,362.9"
                fill="url(#hatch-medium-blue)"
                stroke={c.mediumBlue}
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
                stroke={c.lightBlue}
                strokeWidth="6.4"
                strokeDasharray="12.8,6.4"
                rx="12.8"
                ry="12.8"
            />
            {/* Top section - cyan */}
            <polygon
                points="266.62,328.73 356.1,237.1 356.1,53.82 266.62,145.46"
                fill="url(#hatch-cyan)"
                stroke={c.cyan}
                strokeWidth="6.4"
                strokeDasharray="12.8,6.4"
                strokeLinejoin="round"
            />
            <polygon
                points="356.1,54 114.48,54 25,145.46 266.62,145.46"
                fill="url(#hatch-light-cyan)"
                stroke={c.lightCyan}
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
                stroke={c.teal}
                strokeWidth="6.4"
                strokeDasharray="12.8,6.4"
                rx="12.8"
                ry="12.8"
            />
            {/* Right side elements */}
            <polygon
                points="467.47,452.33 374.48,546.18 374.48,362.9 467.47,269.05"
                fill="url(#hatch-orange)"
                stroke={c.orange}
                strokeWidth="6.4"
                strokeDasharray="12.8,6.4"
                strokeLinejoin="round"
            />
            <polygon
                points="575,452.33 482.01,546.18 482.01,362.9 575,269.05"
                fill="url(#hatch-green)"
                stroke={c.green}
                strokeWidth="6.4"
                strokeDasharray="12.8,6.4"
                strokeLinejoin="round"
            />
        </g>
    );
};

export default XarrayLogo;
