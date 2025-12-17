//
// Fragment shader for Gaussian splats.
// Renders a Gaussian splat within a quad, discarding fragments outside the unit circle.
// Applies an approximate Gaussian falloff based on distance from the center and outputs
// a color modulated by the alpha and Gaussian weight.
//

void main() {
    if (v_splitDirection < 0.0 && gl_FragCoord.x > czm_splitPosition) discard;
    if (v_splitDirection > 0.0 && gl_FragCoord.x < czm_splitPosition) discard;

    float A = -dot(v_vertPos, v_vertPos);
    if (A < -4.) {
        discard;
    }

    float B = exp(A * 4.) * v_splatColor.a ;
    vec3 finalColor = v_splatColor.rgb;

    bool isSelected = false;
    bool isLocked = false;
    uint colorGroupId = 0u;
#if defined(HAS_SPLAT_STATE)
    // Apply selection and lock colors based on state
    // State encoding: bits 0-1 = flags (selected, locked), bits 2-7 = color group ID
    uint vertexState = uint(v_splatState);
    colorGroupId = (vertexState >> 2u) & 0x3Fu; // Extract color group ID (bits 2-7)
    isSelected = (vertexState & 1u) != 0u;
    isLocked = (vertexState & 2u) != 0u;
#endif

    if (u_outlineMaskPass) {
#if defined(HAS_SPLAT_STATE)
        if (!isSelected) {
            discard;
        }
#else
        discard;
#endif
        // 输出软边 alpha mask，保留高斯衰减用于后续滤波
        float alpha = exp(-A * 4.0) * v_splatColor.a;
        out_FragColor = vec4(1.0, 1.0, 1.0, alpha);
        return;
    }

    if (u_outlineRingPass) {
#if defined(HAS_SPLAT_STATE)
        if (!isSelected) {
            discard;
        }
#else
        discard;
#endif
        float gaussian = exp(A * 4.0);
        float ringCenter = clamp(u_outlineAlphaCutoff, 0.2, 0.9);
        float ringWidth = clamp((u_outlineWidth - 0.5) * 0.12, 0.02, 0.3);
        float inner = max(ringCenter - ringWidth, 0.0);
        float outer = min(ringCenter + ringWidth, 1.0);
        float softness = ringWidth * 0.5;
        float ringInner = smoothstep(inner - softness, inner + softness, gaussian);
        float ringOuter = smoothstep(outer - softness, outer + softness, gaussian);
        float ringAlpha = clamp((ringInner - ringOuter) * u_outlineColor.a, 0.0, 1.0);
        out_FragColor = vec4(u_outlineColor.rgb * ringAlpha, ringAlpha);
        return;
    }

#if defined(HAS_SPLAT_STATE)
    if (isLocked) {
        // Locked: multiply by locked color
        finalColor *= u_lockedColor.rgb;
    } else if (isSelected) {
        // Selected: use color from color group or default selection color
        vec4 highlightColor = u_selectedColor; // Default to selection color
        if (colorGroupId > 0u && colorGroupId < uint(u_colorGroupCount)) {
            // Use color from color group array (safe array access)
            highlightColor = u_colorGroupColors[colorGroupId];
        }
        // Mix with highlight color
        finalColor = mix(finalColor, highlightColor.rgb * 0.8, highlightColor.a);
    }
#endif // HAS_SPLAT_STATE

    out_FragColor = vec4(finalColor * B , B);
}
